"""Document existing checkpoints and only locally measured evaluation results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    output = ROOT / 'docs/model_cards'
    output.mkdir(exist_ok=True)
    cards = []
    for path in sorted((ROOT/'checkpoints').glob('patchcore_*.pt')) + sorted((ROOT/'checkpoints/public_efficientad').glob('*/model.ckpt')):
        if path.suffix == '.pt':
            architecture, category = 'patchcore', path.stem.removeprefix('patchcore_')
            provenance = dict(source='existing local checkpoint', training_seed=None, original_training_protocol='not recorded in this checkpoint')
        else:
            architecture, category = 'efficientad', path.parent.name
            provenance = json.loads((path.parent/'source.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024), b''):
                digest.update(block)
        evaluations=[]
        for report in sorted((ROOT/'reports').glob(f'verified_{architecture}_*_{category}_*.json')):
            result=json.loads(report.read_text(encoding='utf-8'))
            if result.get('schema_version') == 2:
                evaluations.append(dict(report=str(report.relative_to(ROOT)), protocol=result['dataset_info'], metrics=result['metrics'], conditions=result['model_info']))
        card=dict(architecture=architecture, category=category, intended_use='Category-specific industrial visual anomaly inspection',
            checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=digest.hexdigest(), provenance=provenance,
            preprocessing='Checkpoint model-specific resize (256x256) and normalization; EfficientAD normalization is internal.',
            score_semantics='Raw scores for ranking; checkpoint-calibrated scores for fixed-threshold decisions; not probabilities.',
            local_evaluations=evaluations, limitations=['No claim of transfer to unrelated conveyor footage.',
                'Unknown/third-party training and validation selection prevents claims of a matched-training benchmark.',
                'No real-video defect accuracy has been established for this model.'])
        (output/f'{architecture}_{category}.json').write_text(json.dumps(card,indent=2,allow_nan=False),encoding='utf-8')
        cards.append(dict(model=architecture,category=category,card=f'{architecture}_{category}.json'))
    (output/'index.json').write_text(json.dumps(cards,indent=2),encoding='utf-8')
    print(f'Wrote {len(cards)} checkpoint model cards to {output}')

if __name__ == '__main__':
    main()
