import mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient()
exp = client.get_experiment_by_name('rossmann')
print('experiment:', exp)
if exp is None:
    print('no experiment named rossmann')
else:
    runs = client.search_runs(exp.experiment_id)
    print('found runs:', len(runs))
    for r in runs:
        print('\n--- run_id:', r.info.run_id, 'status:', r.info.status)
        try:
            arts = client.list_artifacts(r.info.run_id)
            print('artifacts root entries:', [a.path for a in arts])
            def walk(run_id, path=''):
                items = client.list_artifacts(run_id, path)
                for it in items:
                    print(' -', it.path, '(is_dir=%s)'%it.is_dir)
                    if it.is_dir:
                        walk(run_id, it.path)
            walk(r.info.run_id, '')
        except Exception as e:
            print('list artifacts failed:', e)
