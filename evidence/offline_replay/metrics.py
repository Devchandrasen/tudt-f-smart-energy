"""Count detector and dispatch outcomes over a recorded observation horizon."""
import numpy as np


def record_counts(detectors, fault, gate, ungated, end):
    faults = fault[:end]
    rough = detectors[:end, 1]
    jump = detectors[:end, 2]
    alarm = jump | rough
    missed = faults & ~alarm
    full = gate.regime.to_numpy()[:end] == 2
    active = gate.activity.to_numpy()[:end] == 1
    ungated_active = ungated.activity.to_numpy()[:end] == 1
    high = gate.high_error.to_numpy()[:end] == 1
    assert np.array_equal(high, ungated.high_error.to_numpy()[:end] == 1)
    assert np.array_equal(active & high, gate.high_error_automation.to_numpy()[:end] == 1)
    assert np.array_equal(ungated_active & high, ungated.high_error_automation.to_numpy()[:end] == 1)
    return dict(hours=end, corrupt=faults.sum(), noncorrupt=(~faults).sum(),
                joint_tp=(alarm & faults).sum(), joint_fp=(alarm & ~faults).sum(),
                jump_tp=(jump & faults).sum(), jump_fp=(jump & ~faults).sum(),
                rough_tp=(rough & faults).sum(), rough_fp=(rough & ~faults).sum(),
                missed=missed.sum(), missed_full=(missed & full).sum(),
                missed_active=(missed & active).sum(), corrupt_full=(faults & full).sum(),
                corrupt_active=(faults & active).sum(), ungated_corrupt_active=(faults & ungated_active).sum(),
                differing_corrupt_commands=(faults & (gate.command_index.to_numpy()[:end] != ungated.command_index.to_numpy()[:end])).sum(),
                gate_high=(faults & active & high).sum(), ungated_high=(faults & ungated_active & high).sum(),
                active_hours=active.sum())


def aggregate(label, rows, **extra):
    counts={k:sum(int(row[k]) for row in rows) for k in rows[0]}
    assert counts['missed']+counts['joint_tp']==counts['corrupt']
    metrics={}
    for name, num, den in [('recall','joint_tp','corrupt'),('fpr','joint_fp','noncorrupt'),
            ('jump_recall','jump_tp','corrupt'),('jump_fpr','jump_fp','noncorrupt'),
            ('rough_recall','rough_tp','corrupt'),('rough_fpr','rough_fp','noncorrupt'),
            ('full_corrupt','corrupt_full','corrupt'),('active_corrupt','corrupt_active','corrupt'),
            ('full_on_miss','missed_full','missed'),('active_on_miss','missed_active','missed'),
            ('gate_high_pct','gate_high','corrupt'),('ungated_high_pct','ungated_high','corrupt')]:
        metrics[name] = 100*counts[num]/counts[den] if counts[den] else None
    return {'case':label,**counts, **metrics,**extra}
