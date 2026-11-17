"""Doc Protocol table from the 7 logs of eval_doc_protocol.py (one checkpoint).

F1 = per-patch F1 averaged over the patches whose ground truth has at least one tampered pixel, in %.
Rows: F1@0.5 (the reported metric: the learned threshold offset is inside the model, the cut is fixed at 0.5),
then, for reference only, the single sweep threshold that maximises the mean cross-domain F1 (chosen on the test
sets, not deployable).

  python doc_protocol_table.py --logs <dir with <set>.log>
"""
import argparse
import os
import re

SETS = {'DT-Test': 'DocTamperV1-TestingSet', 'DT-FCD': 'DocTamperV1-FCD', 'DT-SCD': 'DocTamperV1-SCD',
        'T-SROIE': 'T-SROIE_test', 'OSTF': 'OSTF_test', 'TPIC-13': 'Tampered-IC13_test',
        'RTM': 'RealTextManipulation_test'}
IN_DOMAIN = ('DT-Test', 'DT-FCD', 'DT-SCD')
CROSS_DOMAIN = ('T-SROIE', 'OSTF', 'TPIC-13', 'RTM')
NUM = r'([0-9]+(?:\.[0-9]+)?)'


def parse(path):
    """(F1@0.5, {t: F1} sweep) of one eval log, in %."""
    text = open(path).read()
    if '=== Forward timing' not in text:
        raise SystemExit(f'{path}: incomplete eval log')
    primary = re.findall(rf'^ F1-score\s*:\s*{NUM}$', text, re.M)
    if len(primary) != 1:
        raise SystemExit(f'{path}: expected one F1-score line, found {len(primary)}')
    rows = re.findall(rf'Threshold {NUM} \(per-patch avg\) ===\n IoU.*\n Precision.*\n Recall.*\n F1\s*:\s*{NUM}%', text)
    sweep = {round(float(t), 4): float(f) for t, f in rows}
    return 100 * float(primary[0]), {t: f for t, f in sweep.items() if 0 < t < 1}


def main(args):
    parsed = {k: parse(os.path.join(args.logs, f'{name}.log')) for k, name in SETS.items()}
    grid = sorted(parsed['RTM'][1])
    if any(sorted(p[1]) != grid for p in parsed.values()):
        raise SystemExit('the 7 logs do not share one sweep grid')
    t = max(grid, key=lambda x: (sum(parsed[k][1][x] for k in CROSS_DOMAIN) / len(CROSS_DOMAIN), -x))
    rows = [('F1@0.5', {k: parsed[k][0] for k in SETS}),
            (f'best cross-domain cut t={t:.2f} (reference)', {k: parsed[k][1][t] for k in SETS})]
    mean = lambda r, ks: sum(r[k] for k in ks) / len(ks)
    print('| row | ' + ' | '.join(SETS) + ' | mean in-domain | mean cross-domain |')
    print('| --- |' + ' ---: |' * (len(SETS) + 2))
    for name, r in rows:
        print(f'| {name} | ' + ' | '.join(f'{r[k]:.2f}' for k in SETS)
              + f' | {mean(r, IN_DOMAIN):.2f} | {mean(r, CROSS_DOMAIN):.2f} |')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--logs', required=True)
    main(parser.parse_args())
