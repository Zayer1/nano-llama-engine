import os
import json

root = r'e:\Antigravity\Antigravity\Projects\Past chat instances'
out_path = r'e:\Antigravity\Antigravity\Projects\all_user_inputs.txt'

with open(out_path, 'w', encoding='utf-8') as out:
    for d in os.listdir(root):
        dir_path = os.path.join(root, d)
        if os.path.isdir(dir_path):
            log_path = os.path.join(dir_path, '.system_generated', 'logs', 'transcript.jsonl')
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data.get('type') == 'USER_INPUT':
                                step = data.get('step_index', '')
                                time = data.get('created_at', '')
                                content = data.get('content', '').replace('\n', ' ').replace('\r', '')
                                out.write(f'[{time}] {d} (step {step}): {content}\n')
                        except:
                            pass
