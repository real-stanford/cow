import os, json, argparse, pprint
from ai2thor.util.metrics import compute_spl
import numpy as np
from scipy import stats


def binomial_ci(successes, trials, alpha=0.05):
    # adapted from: https://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals
    #x is number of successes, n is number of trials

    if successes==0:
        c1 = 0
    else:
        c1 = stats.beta.interval(1-alpha, successes, trials-successes+1)[0]
    if successes==trials:
        c2=1
    else:
        c2 = stats.beta.interval(1-alpha, successes+1, trials-successes)[1]

    return c1, c2

def make_latex_table(cols: str, rows: str, title: str, caption: str, label: str):

    header = \
f"""
\\begin{{table}}
\\centering
\\begin{{tabular}}{{l?{'c'*(len(cols)-1)}}}
\\toprule
"""

    body = ' & '.join(cols) + '\\\\\\midrule\n'
    for row in rows:
        body += ' & '.join(row) + '\\\\\n'


    footer = \
f"""
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{{title}}} {caption}}}
\\label{{{label}}}
\\end{{table}}
"""

    return header + body + footer


def results_robo(result_dir):

    denom = 0
    num = 0

    obj_counts = {}

    a = {}
    b = {}

    episode_results = []

    total_actions = 0
    successful_actions = 0

    episode_results_per_cat = {}
    spl_per_cat = {}
    success_per_cat = {}

    for name in os.listdir(result_dir):
        denom += 1
        target = name.split('.')[0].split('_')[3]

        if target not in obj_counts:
            obj_counts[target] = {'num' : 0, 'denom' : 0}
            episode_results_per_cat[target] = []

        with open(f'{result_dir}/{name}', 'r') as f:
            obj = json.loads(f.read())
            a[name] = obj['episode_result']['success']

            obj['episode_result']['vision_fail'] = 0
            obj['episode_result']['exploration_fail'] = 0
            obj['episode_result']['planning_fail'] = 0


            if obj['episode_result']['success'] == 1:
                # print(name)
                num += 1
                obj_counts[target]['num'] += 1
            else:
                if 'vision_error_in_case_of_fail' in obj['episode_metrics']:
                    target_seen = obj['episode_metrics']['vision_error_in_case_of_fail']
                    if len(obj['episode_metrics']['trajectory']) == 251 and not target_seen:
                        obj['episode_result']['exploration_fail'] = 1
                    else:
                        if len(obj['episode_metrics']['trajectory']) == 251:
                            obj['episode_result']['planning_fail'] = 1
                        else:
                            obj['episode_result']['vision_fail'] = 1

            obj_counts[target]['denom'] += 1
            episode_results.append(obj['episode_result'])
            episode_results_per_cat[target].append(obj['episode_result'])

            total_actions += len(obj['episode_metrics']['actions_taken'])
            for ac in obj['episode_metrics']['actions_taken']:
                if ac['success']:
                    successful_actions += 1

    if len(episode_results) == 0:
        print(result_dir)

    ep_len = sum([len(em["path"]) for em in episode_results]) / len(episode_results)
    success = sum([r["success"] for r in episode_results]) / len(episode_results)
    failure = 1 - success

    vision_fail = sum([r["vision_fail"] for r in episode_results]) / (len(episode_results))
    exp_fail = sum([r["exploration_fail"] for r in episode_results]) / (len(episode_results))
    planning_fail = sum([r["planning_fail"] for r in episode_results]) / (len(episode_results))

    spl = compute_spl(episode_results)

    for target in episode_results_per_cat:
        spl_per_cat[target] = compute_spl(episode_results_per_cat[target])
        success_per_cat[target] = np.mean([int(r["success"]) for r in episode_results_per_cat[target]]).item()

    return {
        'spl': spl,
        'success': success,
        'failure': failure,
        'vision_failure_fraction': vision_fail,
        'exploration_failure_fraction': exp_fail,
        'planning_failure_fraction': planning_fail,
        'action_success': successful_actions/total_actions,
        'spl_category': spl_per_cat,
        'success_category': success_per_cat,
        'samples':  len(episode_results),
    }

def results_habitat(result_dir):
    denom = 0
    num = 0

    obj_counts = {}
    spl_sum = {}
    successful_actions_sum = {}
    success_sum = {}

    all_spl = []

    success_count = 0
    to_term_count = 0
    early_stop_fail_count = 0

    for name in os.listdir(result_dir):
        if 'json' in name:
            denom += 1
            target = '_'.join(name.split('.')[0].split('_')[:-1])

            if target not in obj_counts:
                obj_counts[target] = 0
                spl_sum[target] = []
                successful_actions_sum[target] = []
                success_sum[target] = []

            with open(f'{result_dir}/{name}', 'r') as f:
                obj = json.loads(f.read())
                obj_counts[target] += 1
                spl_sum[target].append(obj['spl'])
                all_spl.append(obj['spl'])
                successful_actions_sum[target].append(obj['successful_actions'])
                success_sum[target].append(obj['success'])

                if obj['success']:
                    success_count += 1
                else:
                    if 'total_actions' in obj:
                        if obj['total_actions'] == 250:
                            to_term_count += 1
                        else:
                            # print(name)
                            early_stop_fail_count += 1

    total_evals = 0
    total_spl = 0
    total_successful_actions = 0
    total_success = 0

    spl_per_cat = {}
    success_per_cat = {}

    for target in sorted(obj_counts):

        total_evals += obj_counts[target]
        total_spl += np.sum(spl_sum[target])
        total_successful_actions += np.sum(successful_actions_sum[target])
        total_success += np.sum(success_sum[target])

        assert len(spl_sum[target]) == obj_counts[target]

        spl_per_cat[target] = np.mean(spl_sum[target])
        success_per_cat[target] = np.mean(success_sum[target])

    spl_balanced = np.mean(list(spl_per_cat.values()))
    success_balanced = np.mean(list(success_per_cat.values()))

    return {
        'spl': total_spl / total_evals,
        'success': total_success / total_evals,
        'action_success': total_successful_actions / total_evals,
        'samples': total_evals,
        'spl_balanced': spl_balanced,
        'success_balanced': success_balanced,
        'success_category': success_per_cat,
        'spl_category': spl_per_cat,
        'samples_cat': obj_counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Aggrigate results.')

    parser.add_argument('--result-dir', action='store', type=str, required=True)
    parser.add_argument('--hab', action='store_true', default=False)

    args = parser.parse_args()

    if args.hab:
        pprint.pprint(results_habitat(args.result_dir))
    else:
        pprint.pprint(results_robo(args.result_dir))