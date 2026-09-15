"""Read-only ROS1 GNSS extraction; no positioning or satellite propagation.

Types are registered from pinned official .msg files and checked against each
selected connection's ROS1 MD5. A bag's embedded definition is never executed.
"""
from __future__ import annotations

import argparse
import dataclasses
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

ROOT = Path(__file__).resolve().parents[2] / 'configs'
MESSAGE_ROOT = Path(__file__).resolve().parents[2] / 'third_party' / 'gnss_comm' / 'msg'
COMM = ROOT.parent / 'gnss_comm-main'
C = 299792458.0
GPS_EPOCH_UNIX_S = 315964800
TOPICS = {
    '/ublox_driver/range_meas': 'GnssMeasMsg',
    '/ublox_driver/ephem': 'GnssEphemMsg',
    '/ublox_driver/glo_ephem': 'GnssGloEphemMsg',
    '/ublox_driver/receiver_pvt': 'GnssPVTSolnMsg',
}
OPTIONAL = {
    '/ublox_driver/iono_params': 'StampedFloat64Array',
    '/ublox_driver/time_pulse_info': 'GnssTimePulseInfoMsg',
}
ARRAY_FIELDS = ('freqs', 'CN0', 'LLI', 'code', 'psr', 'psr_std',
                'cp', 'cp_std', 'dopp', 'dopp_std', 'status')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')


def official_typestore():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text('utf-8'))
    types = {}
    for entry in manifest['message_definitions']:
        path = MESSAGE_ROOT / entry['filename']
        if sha(path) != entry['sha256']:
            raise ValueError(f'MESSAGE_SOURCE_HASH_CONFLICT: {path}')
        types.update(get_types_from_msg(path.read_text('utf-8'),
                                       f'gnss_comm/msg/{path.stem}'))
    store = get_typestore(Stores.ROS1_NOETIC)
    store.register(types)
    return store


def checked_connections(reader, store, optional=False):
    mapping = TOPICS | (OPTIONAL if optional else {})
    selected = []
    for conn in reader.connections:
        if conn.topic not in mapping:
            continue
        expected = 'gnss_comm/msg/' + mapping[conn.topic]
        if conn.msgtype != expected:
            raise ValueError(f'TOPIC_TYPE_CONFLICT: {conn.topic}: {conn.msgtype}')
        _, md5 = store.generate_msgdef(expected, ros_version=1)
        if conn.digest != md5:
            raise ValueError(f'MESSAGE_MD5_CONFLICT: {conn.topic}: '
                             f'bag={conn.digest} official={md5}')
        selected.append(conn)
    return selected


def gpst_ns(week, tow):
    """GPST calendar on Unix origin, NOT UTC Unix nanoseconds."""
    if int(week) < 0 or not math.isfinite(float(tow)) or not 0 <= float(tow) < 604800:
        raise ValueError('INVALID_GPS_WEEK_TOW')
    # Preserve subsecond TOW without adding it to a ~1e9 floating-point epoch.
    fractional_ns = int((Decimal(str(tow)) * 1_000_000_000).to_integral_value(
        rounding=ROUND_HALF_EVEN))
    return (GPS_EPOCH_UNIX_S + int(week) * 604800) * 1_000_000_000 + fractional_ns


def sat_identity(sat):
    # Exact boundaries in gnss_constant.hpp + satsys, not u-blox native gnssId.
    for offset, count, system in ((0, 32, 'GPS'), (32, 27, 'GLO'),
                                  (59, 38, 'GAL'), (97, 63, 'BDS')):
        if offset < sat <= offset + count:
            return system, sat - offset
    return 'UNKNOWN', None


def plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)
                if not f.name.startswith('__')}
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def signal_rows(msg, bag_ns, message_index):
    rows = []
    for obs_index, obs in enumerate(msg.meas):
        lengths = {k: len(getattr(obs, k)) for k in ARRAY_FIELDS}
        n = lengths['freqs']
        # Reject structural inconsistencies instead of aligning arrays by guess.
        if any(length != n for length in lengths.values()):
            raise ValueError(f'SIGNAL_ARRAY_LENGTH_CONFLICT: {lengths}')
        system, prn = sat_identity(int(obs.sat))
        epoch = gpst_ns(obs.time.week, obs.time.tow)
        for i in range(n):
            values = {k: plain(getattr(obs, k)[i]) for k in ARRAY_FIELDS}
            f, d, sd = values['freqs'], values['dopp'], values['dopp_std']
            usable = f is not None and f > 0 and d is not None and sd is not None and sd > 0
            rows.append({
                'row_id': f'{message_index}:{obs_index}:{i}',
                'bag_record_ns': bag_ns, 'gps_week': int(obs.time.week),
                'gps_tow_s': float(obs.time.tow), 'gpst_calendar_ns': epoch,
                'sat': int(obs.sat), 'system': system, 'prn': prn,
                'signal_index': i, **values,
                'range_rate_raw_mps': -C / f * d if usable else None,
                'range_rate_std_mps': C / f * sd if usable else None,
                'doppler_numeric_available': bool(usable),
                'pseudorange_valid_bit': bool(int(values['status']) & 1),
                'doppler_valid_status_bit': 'NOT_DEFINED_IN_OFFICIAL_MESSAGE',
                'role': 'OBSERVATION_ONLY_NOT_MODEL_READY',
            })
    return rows


def pvt_row(msg, bag_ns):
    result = plain(msg)
    lat, lon = np.deg2rad([msg.latitude, msg.longitude])
    a, e2 = 6378137.0, 6.69437999014e-3
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    p = np.array([(n + msg.altitude) * np.cos(lat) * np.cos(lon),
                  (n + msg.altitude) * np.cos(lat) * np.sin(lon),
                  (n * (1 - e2) + msg.altitude) * np.sin(lat)])
    enu = np.array([msg.vel_e, msg.vel_n, -msg.vel_d])
    rotation = np.array([[-np.sin(lon), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
                         [ np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
                         [0, np.cos(lat), np.sin(lat)]])
    ve = rotation @ enu
    quality = ('INVALID_NO_FIX' if not msg.valid_fix else
               'RTK_FIXED' if msg.carr_soln == 2 else
               'RTK_FLOAT' if msg.carr_soln == 1 else 'VALID_NON_RTK_FIXED')
    return {**result, 'bag_record_ns': bag_ns,
            'gpst_calendar_ns': gpst_ns(msg.time.week, msg.time.tow),
            'gnss_ts_ns_official_float_compat': int(
                (GPS_EPOCH_UNIX_S + int(msg.time.week)*604800 + float(msg.time.tow))*1e9),
            **dict(zip(('ecef_px', 'ecef_py', 'ecef_pz'), p.tolist())),
            **dict(zip(('enu_vx', 'enu_vy', 'enu_vz'), enu.tolist())),
            **dict(zip(('ecef_vx', 'ecef_vy', 'ecef_vz'), ve.tolist())),
            'quality_class': quality,
            'posthoc_fixed_eligible': bool(msg.valid_fix and msg.carr_soln == 2),
            'role': 'POSTHOC_REFERENCE_DATA_QUALITY_REFERENCE'}


def inventory(bag, output):
    store = official_typestore()
    with Reader(bag) as reader:
        chosen = checked_connections(reader, store, optional=True)
        found = {c.topic for c in chosen}
        report = {'bag': str(bag.resolve()), 'size_bytes': bag.stat().st_size,
                  'start_bag_ns': reader.start_time, 'end_bag_ns': reader.end_time,
                  'duration_ns': reader.duration, 'message_count': reader.message_count,
                  'gnss_md5_check': 'PASS', 'missing_core_topics': sorted(TOPICS.keys() - found),
                  'body_messages_deserialized': 0,
                  'topics': [{'topic': c.topic, 'msgtype': c.msgtype, 'md5': c.digest,
                              'message_count': c.msgcount,
                              'role': ('POSTHOC_REFERENCE' if c.topic.endswith('receiver_pvt') else
                                       'GNSS_EXTRACTION' if c in chosen else 'NOT_USED_BY_DOPPLER_SOLVER')}
                             for c in reader.connections]}
    save_json(output, report)
    return report


def extract(bag, out, week, tow, duration=30.0, optional=False):
    if not math.isfinite(duration) or not 0 < duration <= 30:
        raise ValueError('PILOT_DURATION_MUST_BE_IN_0_30_SECONDS')
    if out.exists():
        raise FileExistsError('Output must be a new directory; existing extracts are never overwritten')
    start = gpst_ns(week, tow)
    stop = start + int(round(duration * 1e9))
    store = official_typestore()
    out.mkdir(parents=True)
    (out / 'observations').mkdir()
    (out / 'reference').mkdir()
    count = {'raw_signal_rows': 0, 'pvt_rows': 0, 'ephem_messages': 0,
             'glo_ephem_messages': 0, 'optional_messages': 0}
    message_count = 0
    raw_writer = pvt_writer = None
    last_times = {}
    timing = {'raw_nonmonotone': 0, 'pvt_nonmonotone': 0}
    outputs = []
    try:
        with Reader(bag) as reader, (out/'observations/ephemerides.jsonl').open('x', encoding='utf-8') as ef, \
                (out/'observations/optional.jsonl').open('x', encoding='utf-8') as of:
            selected = checked_connections(reader, store, optional)
            for conn, timestamp, data in reader.messages(connections=selected):
                message_count += 1
                msg = store.deserialize_ros1(data, conn.msgtype)
                if conn.topic.endswith('/range_meas'):
                    rows = [r for r in signal_rows(msg, timestamp, message_count)
                            if start <= r['gpst_calendar_ns'] < stop]
                    if rows:
                        t = rows[0]['gpst_calendar_ns']
                        if t < last_times.get('raw', t): timing['raw_nonmonotone'] += 1
                        last_times['raw'] = t
                        # Fixed schema prevents a first all-null invalid signal from defining a null column.
                        table = pa.Table.from_pylist(rows, schema=RAW_SCHEMA)
                        if raw_writer is None:
                            path = out/'observations/raw_gnss.parquet'
                            raw_writer = pq.ParquetWriter(path, RAW_SCHEMA, compression='zstd')
                            outputs.append(path)
                        raw_writer.write_table(table)
                        count['raw_signal_rows'] += len(rows)
                elif conn.topic.endswith('/receiver_pvt'):
                    t = gpst_ns(msg.time.week, msg.time.tow)
                    if start <= t < stop:
                        if t < last_times.get('pvt', t): timing['pvt_nonmonotone'] += 1
                        last_times['pvt'] = t
                        table = pa.Table.from_pylist([pvt_row(msg, timestamp)])
                        if pvt_writer is None:
                            path = out/'reference/rtk_reference.parquet'
                            pvt_writer = pq.ParquetWriter(path, table.schema, compression='zstd')
                            outputs.append(path)
                        pvt_writer.write_table(table)
                        count['pvt_rows'] += 1
                elif conn.topic.endswith(('/ephem', '/glo_ephem')):
                    # Navigation records are retained across the bag for ephemeris availability;
                    # their arrival times remain explicit for a causal pilot join.
                    ef.write(json.dumps({'topic': conn.topic, 'bag_record_ns': timestamp,
                                         'message': plain(msg)}, allow_nan=False) + '\n')
                    count['glo_ephem_messages' if conn.topic.endswith('/glo_ephem') else 'ephem_messages'] += 1
                else:
                    # Auxiliary metadata only; never camera, IMU, or receiver solution as candidate input.
                    of.write(json.dumps({'topic': conn.topic, 'bag_record_ns': timestamp,
                                         'message': plain(msg)}, allow_nan=False) + '\n')
                    count['optional_messages'] += 1
    except Exception as exc:
        save_json(out/'extraction_failure.json', {
            'status':'PARTIAL_EXTRACTION_NOT_VALIDATED', 'error_type':type(exc).__name__,
            'error':str(exc), 'counts_so_far':count, 'source_bag':str(bag.resolve()),
            'positioning_experiments':0,
            'action':'Keep partial directory; investigate exact cause before a new output directory'})
        raise
    finally:
        if raw_writer: raw_writer.close()
        if pvt_writer: pvt_writer.close()
    outputs += [out/'observations/ephemerides.jsonl', out/'observations/optional.jsonl']
    report = {'source_bag': str(bag.resolve()), 'source_size_bytes': bag.stat().st_size,
              'extractor_sha256': sha(__file__), 'source_manifest_sha256': sha(ROOT/'source_manifest.json'),
              'interval_identity': 'GPST_CALENDAR_NS_HALF_OPEN', 'start': start, 'stop': stop,
              'duration_s': duration, 'counts': count, 'timing': timing,
              'selected_gnss_messages_deserialized': message_count,
              'camera_imu_messages_deserialized': 0, 'positioning_experiments': 0,
              'satellite_propagation': 'NOT_RUN', 'forward_qa': 'NOT_RUN',
              'ephemeris_scope': 'ALL_NAVIGATION_MESSAGES; causal availability join required',
              'io_note': 'Bag chunks can contain camera bytes even though no camera message is deserialized.',
              'outputs': {str(p.relative_to(out)): sha(p) for p in outputs},
              'status': 'RAW_EXTRACTION_COMPLETE' if count['raw_signal_rows'] else 'NO_RAW_ROWS_IN_REQUESTED_INTERVAL'}
    save_json(out/'extraction_manifest.json', report)
    return report


RAW_SCHEMA = pa.schema([
    ('row_id', pa.string()), ('bag_record_ns', pa.int64()), ('gps_week', pa.int64()),
    ('gps_tow_s', pa.float64()), ('gpst_calendar_ns', pa.int64()), ('sat', pa.int64()),
    ('system', pa.string()), ('prn', pa.int64()), ('signal_index', pa.int64()),
    *[(k, pa.int64() if k in ('LLI','code','status') else pa.float64()) for k in ARRAY_FIELDS],
    ('range_rate_raw_mps', pa.float64()), ('range_rate_std_mps', pa.float64()),
    ('doppler_numeric_available', pa.bool_()), ('pseudorange_valid_bit', pa.bool_()),
    ('doppler_valid_status_bit', pa.string()), ('role', pa.string())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('inventory')
    check.add_argument('--bag', type=Path, required=True)
    check.add_argument('--output', type=Path, required=True)
    sub = commands.add_parser('extract')
    sub.add_argument('--bag', type=Path, required=True)
    sub.add_argument('--out', type=Path, required=True)
    sub.add_argument('--week', type=int, required=True)
    sub.add_argument('--tow', type=float, required=True)
    sub.add_argument('--duration', type=float, default=30)
    sub.add_argument('--optional', action='store_true')
    args = parser.parse_args()
    if not args.bag.is_file(): parser.error('Bag file is absent; this program never downloads data')
    if args.command == 'inventory':
        if args.output.exists(): parser.error('Inventory output already exists')
        result = inventory(args.bag, args.output)
    else:
        result = extract(args.bag, args.out, args.week, args.tow, args.duration, args.optional)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
