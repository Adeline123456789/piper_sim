"""Place complete task recordings in the user's four outcome directories."""
import json
from pathlib import Path
import shutil


def archive_episode(root, session, episode):
    root = Path(root)
    source = root/'proprio_stats'/session/episode
    metadata = json.loads((source/'metadata.json').read_text())
    domain = metadata['domain']
    if domain == 'real':
        outcome = metadata.get('execution_outcome') or {}
        result = outcome.get('physical_task_success')
        if result is None:
            raise ValueError('Real task outcome must be observed before archiving')
        branch = 'complete_real/successed' if result else 'complete_real/failed'
    elif domain == 'sim':
        branch = 'complete_sim/succeed' if metadata['success'] else 'complete_sim/failed'
    else:
        raise ValueError('Unknown recording domain')
    target = root/branch/f'{session}_{episode}'
    target.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, target/'proprio_stats')
    shutil.copytree(root/'observations'/session/episode/'videos', target/'videos')
    (target/'outcome.json').write_text(json.dumps(dict(domain=domain,
        success=bool(metadata['success']) if domain=='sim' else bool(result),
        source_session=session, source_episode=episode,
        policy_training_eligible=metadata['policy_training_eligible']), indent=2)+'\n')
    return target
