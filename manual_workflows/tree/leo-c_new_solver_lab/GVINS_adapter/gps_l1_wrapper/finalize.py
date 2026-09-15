"""Verify and document completed fixed-state QA; no physical refits."""
from pathlib import Path
import json,csv,hashlib,math
from datetime import datetime,timezone
from dataclasses import fields
import wrapper

OUT=Path(__file__).resolve().parent
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
summary=read('qa_summary.json'); manifest=read('input_manifest.json'); tables=read('tables.json')
firewall=read('pseudorange_firewall_test.json'); leaf=read('native_leaf_identity_test.json')
assert firewall['status']=='PASS' and not firewall['wrapper_io_events']
assert summary['jacobian_all_pass'] and summary['qr_forward_max_relative']<1e-5
assert summary['nested_all_bitwise_equal'] and leaf['optimizer_calls']==0
assert sha(OUT/'wrapper.py')==read('qa_execution_lock.json')['wrapper_sha256']
assert sha(OUT/'qa.py')==read('qa_execution_lock.json')['qa_sha256']
input_checks=[]
for item in manifest['inputs']+manifest['native_sources']+[manifest['physics_source']]:
    assert sha(item['path'])==item['sha256'],item['path']
    input_checks.append(item)
for a in read('table_export_verification.json'):
    with (OUT/(a['table']+'.csv')).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==len(tables[a['table']])==a['rows']
    for found,expected in zip(rows,tables[a['table']]):
        for key,value in expected.items():
            s=found[key]
            if value is None: assert s==''
            elif isinstance(value,bool): assert s.lower()==str(value).lower()
            elif isinstance(value,(int,float)): assert float(s)==value
            else: assert s==str(value)

ephprops={f.name:{'type':'integer' if f.name in ['sat','toe_ns','toc_ns','health'] else 'number'} for f in fields(wrapper.Ephemeris)}
props={'receive_gpst_ns':{'type':'integer'},'satellite':{'type':'integer'},'system':{'const':'GPS'},'signal':{'const':'CODE_L1C'},'frequency_hz':{'const':1575420000.0},'doppler_hz':{'type':'number'},'ephemeris':{'type':'object','additionalProperties':False,'required':list(ephprops),'properties':ephprops}}
save('candidate_input_schema.json',{'$schema':'https://json-schema.org/draft/2020-12/schema','title':'GPS L1 candidate observation; state is a separate physical parameter argument','type':'object','additionalProperties':False,'required':list(props),'properties':props,
    '$comment':'Only prelocked causal healthy ephemeris records. Candidate state uses native p0/v/b0/bdot layout and integer time origin. No pseudorange, reference files, PVT/RTK fields, QA clock estimates, uncertainty weights or oracle inputs.'})

decision='GVINS_GPS_L1_WRAPPER_STATE_RANGE_OR_SCALE_LIMITED'
next_task='GVINS_GPS_L1_FROZEN_CLOCK_PARAMETER_CONTRACT_REVIEW_V1'
criteria={
 'pseudorange_firewall':'PASS', 'reference_field_and_file_firewall':'PASS_WITH_QA_STATE_INJECTION_IDENTITY',
 'physics_prediction':'PASS_EQUATIONS_WITH_TRANSMIT_TIME_DIFFERENCE_QUANTIFIED',
 'light_time_180_rows':'PASS','satellite_clock_and_earth_rotation':'PASS',
 'analytic_jacobian_vs_fd':'PASS','clock_range_and_native_qualification':'LIMITED_BETA0_20_MPS_RULE',
 'projected_robust_refined_leaf_identity':'PASS_FIXED_STATE_LEAVES_ONLY',
 'native_weighting_unchanged':'PASS'}
save('decision.json',{'task':'GVINS_GPS_L1_MEASUREMENT_WRAPPER_IMPLEMENTATION_AND_FROZEN_STATE_QA_V1','run_id':summary['run_id'],'source_run_id':manifest['source_run_id'],'decision':decision,
  'implementation_and_fixed_state_qa':'COMPLETE','frozen_egshs_pilot_authorized_by_this_result':False,
  'primary_limitation':'All thirteen clock-bearing native candidates are algebraically unbounded but reject abs(b0)>20 m/s; observed posthoc common rate is about81.5 m/s. GIR also applies this rule to proposed steps.',
  'secondary_limitation':'Constant/affine clock remainder exceeds source raw-sigma/post-epoch-clock residual scale; cascade drift condition0.02 m/s2 is below the observed affine slope.',
  'success_criteria':criteria,'PSEUDORANGE_USED_BY_CANDIDATE':False,'RTK_PVT_USED_BY_CANDIDATE':False,'WEIGHTING_CHANGED':False,'M0_M14_FITS':0,'EGSHS_RUNS':0,'POSITIONING_EXPERIMENTS':0,
  'next_task':next_task,'next_task_executed':False,'utc':datetime.now(timezone.utc).isoformat()})

report=f'''# GPS L1 measurement wrapper与固定状态QA

任务：GVINS_GPS_L1_MEASUREMENT_WRAPPER_IMPLEMENTATION_AND_FROZEN_STATE_QA_V1  
run_id：{summary['run_id']}  
前置run：{manifest['source_run_id']}  
正式decision：**{decision}**

## 完成结果

确定性GNSS物理封装及固定状态检查已经完成。伪距与参考字段隔离、180行光行时、卫星直接解析导数、预测及Jacobian检查通过。当前不能据此运行冻结EGSHS：原生物理资格要求abs(b0)≤20 m/s，前置QA的公共距离率约81.5 m/s，13个带clock候选均存在这一合同限制。限制来自原有资格语义，状态方程本身的b0没有有限优化器箱形边界；本轮没有修改任何参数或规则。

输出属于DETERMINISTIC_GNSS_MEASUREMENT_PHYSICS_WRAPPER。没有定位结果，没有新接收机状态，没有修改EGSHS或selector。M0_M14_FITS=0、EGSHS_RUNS=0、POSITIONING_EXPERIMENTS=0。

## 来源与数据分母

复用V2的锁定区间：GPST-labelled week2134，TOW[83928.102,83958.102)秒。canonical 1 Hz输入为30个实际测量历元、6颗GPS L1 C/A卫星、180行，PRN为2、5、6、13、29、30。原1 Hz文件还含其他系统/信号，这里按原锁定GPS行和星历关联提取，没有重新按残差筛选。native 10 Hz没有新增预测；已保存的300历元公共clock序列仅用于允许的posthoc时钟形式分析。

原V2输入逐项与final_output_seal核对，必要冻结runtime源码与GVINS_adapter/source_manifest.json中的SHA核对，并在完成后再次验证。绑定的是task21实际独立runtime；读取范围只含传播和必要科学接口，没有读取论文或旧定位数据。输入和实际源码清单见input_manifest.json。原bag没有重新扫描或复制。

## 无伪距预测路径与数据隔离

wrapper接口只接受观测时间、GPS L1信号、Doppler、广播星历和外部显式给定的candidate state。其内部从固定0.075秒光行时初值出发，按当前candidate position迭代卫星发射位置及地球自转；预测包括完整retarded range-rate分母、卫星clock多项式及相对论导数、原candidate公共clock列。接收机clock不参与伪距式时间初始化，也没有扣除前置QA的clock序列。

接口严格拒绝额外字段；全部禁止字段删除后180个固定状态预测逐值相同。实际wrapper调用期间的Python文件/网络访问审计记录为空。RTK/PVT仅由独立QA evaluator读取，并作为已知物理状态值传入固定状态试验；该安排不构成实际候选初始化方案。精确的允许字段和测试范围分别见candidate_input_schema.json、pseudorange_firewall_test.json。

## 预测和导数结果

| 检查 | 实际结果 |
|---|---:|
| 主180行光行时 | 180/180收敛；每行5次迭代 |
| 保存精度下最大光行时闭合误差 | 0秒；不表示无限精度误差为零 |
| 新预测−V2原保存预测，中位数 | +0.000121529 m/s |
| 新预测−V2原保存预测，RMS | 0.000144974 m/s |
| 新预测−V2原保存预测，绝对p95 / maximum | 0.000239984 / 0.000240428 m/s |
| 与V2对齐发射时刻后最大方程差 | 1.137×10⁻¹² m/s |
| 新发射时刻−原伪距路径，中位数 | +0.002209870秒 |
| 直接卫星解析导数−0.1秒位置差分，最大3D差 | 3.147×10⁻⁷ m/s |
| 解析Jacobian−中心差分，全部列最大绝对差 | 5.418×10⁻⁹，按各列物理导数单位 |
| QR正向响应最大相对差 | 2.115×10⁻¹⁰ |
| QR逆系数响应最大相对差 | 4.182×10⁻⁷ |

旧路径以观测伪距给出发射初值，并作卫星钟差转换；新路径按candidate几何光行时求解，两者的发射时刻并不完全相同。完整差异如上保存，不能写成原保存预测逐位相等。相同发射时刻下的方程一致性独立通过，原始路径差远小于本pilot原始sigma中位数0.09743 m/s，未通过拟合修正消除这项差异。

Jacobian采用解析链式法则的前向自动微分，覆盖光行时随接收机位置变化、卫星速度随发射时间变化、自转、LOS、分母及clock项。三组固定状态分别检查物理步长和缩放坐标步长，共48列级记录；各模型的有限性、秩、列范数和光行时另有45条记录。缩放坐标只用于数值QA，没有安装进原solver。秩诊断没有修改旧阈值。

M0/M4与M1/M3在动态velocity=0时，三组固定位置/clock下预测均逐位相同。clock线性列保持+1、+(t-t0)，残差列保持-1、-(t-t0)，与native bias_design_matrix逐值一致。投影beta仍为单独线性变量。原有static、CTD/robust、BE与GIR/cascade消费端的10条固定状态调用路径经临时内存叶函数替换检查，预测/Jacobian差均为零，之后还原8个函数绑定。此项没有启动优化器，未声称完成全候选、训练子集或生产加载器集成。

## clock范围与时间形式

原生M1/M3/M5/M8有constant clock，其余clock-bearing动态/精化状态包含affine clock。所有b0均为m/s、bdot均为m/s²，无需再次乘c。M14从M0初始化，但其原生精化后的状态为完整8维CTD，不能称为最终仍然static。完整逐模型映射见clock_and_unit_mapping.md。

普通LM/BE没有有限b0箱形优化边界，但候选finalize统一调用physical_plausibility，abs(b0)>20 m/s会失败；GIR还在提议步骤上调用这项检查。因而不能利用“代数上可表达”绕过原资格。M0/M4本来没有公共clock列，记录为该量级不可由其clock表达，未将缺列视为代码错误。一般real bdot上限0.1 m/s²，cascade额外要求abs(bdot)<0.02 m/s²，均保持原样。

对已保存的逐epoch公共offset作普通最小二乘constant/affine拟合，仅属于posthoc物理分析：

| 序列 | 时间形式 | b0(m/s) | bdot(m/s²) | 余项RMS(m/s) | 绝对余项p95(m/s) |
|---|---|---:|---:|---:|---:|
| Native 300 epochs | Constant | 81.475370 | 0 | 0.459044 | 0.868385 |
| Native 300 epochs | Affine | 80.854493 | 0.041530 | 0.285247 | 0.513015 |
| Canonical 30 epochs | Constant | 81.444674 | 0 | 0.546929 | 0.974016 |
| Canonical 30 epochs | Affine | 80.672883 | 0.050116 | 0.333117 | 0.513897 |

在主1 Hz序列上，affine捕获的变化多于constant，但余项RMS0.3331 m/s仍高于sigma中位数0.09743 m/s和逐epoch去公共clock后残差RMS0.10750 m/s。该比较没有增加适配门限，说明不能直接把前置“每个epoch一个clock offset”的一致性外推成“原constant/affine clock已充分适配”。这里的公共offset还含共享参考/星历误差，并非独立振荡器测量。1 Hz斜率也超过原cascade的0.02条件。

## 权重与适用边界

原基础损失继续为m/s残差平方；Cauchy、BE投影先验和GIR几何权重保持原实现。raw Doppler sigma仅用于QA量级比较，没有加入candidate schema或loss；WEIGHTING_CHANGED=false。表格技能用于9份CSV的单位、分母与精确数值往返检查，没有生成工作簿。

RAWX仍是receiver-local、近似GPS对齐的时间标签。约2.21毫秒的新旧发射时刻差不能自动被解释为精确接收机clock bias测量；本轮没有估计绝对接收机时间偏置。PVT速度来自同一设备内部导航解，相关性限制继承V2。本轮固定状态QA没有位置误差评价，不证明任何模型的定位表现或clock结构充分性。

## 唯一后续方向

**{next_task}**：针对真实GPS公共clock量级与冻结beta0/漂移资格、原时钟时间形式之间的适配合同进行限定核对，由明确授权决定后续边界。当前停止在STATE_MODEL_OR_PARAMETER_CONTRACT_REVIEW，没有放宽20 m/s、0.1或0.02限制，没有预扣除clock，没有启动M0–M14或EGSHS。
'''
with (OUT/'report.md').open('x',encoding='utf-8') as f:f.write(report)

required=['wrapper.py','wrapper_contract.md','candidate_input_schema.json','pseudorange_firewall_test.json','clock_and_unit_mapping.md','frozen_state_range_compatibility.csv','clock_affine_adequacy.csv','prediction_equivalence_qa.csv','jacobian_qa.csv','nested_state_identity_qa.csv','weight_identity.md','report.md','decision.json']
assert all((OUT/n).is_file() for n in required)
save('verification.json',{'status':'PASS_IMPLEMENTATION_QA_WITH_NATIVE_CLOCK_CONTRACT_LIMITATION','utc':datetime.now(timezone.utc).isoformat(),
    'input_and_native_source_hashes_rechecked':input_checks,'wrapper_matches_pre_QA_hash':True,'qa_matches_pre_QA_hash':True,
    'output_table_exact_roundtrip':True,'success_criteria':criteria,'native_leaf_test':leaf,'required_deliverables':required,
    'no_nonlinear_candidate_fits':True,'no_manuscript_changes':True,'no_selector_or_weight_changes':True,
    'note':'Physical prediction and derivative QA completed; not a READY decision for full frozen candidate execution.'})
save('output_manifest.json',{'run_id':summary['run_id'],'utc':datetime.now(timezone.utc).isoformat(),'files':[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file()],'self_excluded':True})
terminal=dict(TASK_STATUS=decision,TOTAL_REPORT=str(OUT/'report.md'),PSEUDORANGE_USED_BY_CANDIDATE='false',RTK_PVT_USED_BY_CANDIDATE='false',
    LIGHT_TIME_CONVERGENCE='180/180; 5 iterations each',PREDICTION_EQUIVALENCE='V2 saved RMS difference0.000144974 m/s; same-tx max1.137e-12 m/s',
    JACOBIAN_QA='PASS; max absolute5.418e-9; QR forward relative2.115e-10',CLOCK_RANGE_COMPATIBILITY='LIMITED:13 clock-bearing candidates violate native abs(b0)<=20 m/s',
    CLOCK_AFFINE_ADEQUACY='1Hz b0=80.672883 m/s; bdot=0.0501163 m/s2; remainder RMS0.333117 m/s',
    NESTED_STATE_IDENTITY='M0/M4 and M1/M3: bitwise equal at v=0',WEIGHTING_CHANGED='false',M0_M14_FITS=0,EGSHS_RUNS=0,NEXT_TASK=next_task)
for k,v in terminal.items():print(f'{k}={v}')
