import argparse
import logging
from src.utils.config import load_config
from src.models.classifiers import MiniRocketXGBClassifier, MultiRocketXGBClassifier
from src.localization.localizer import Localizer
from src.pipelines.two_stage import TwoStagePipeline
from src.evaluation.evaluator import Evaluator
from src.data.preprocessor import load_and_preprocess_data, run_scrape, run_download, run_preprocess, load_preprocessed_dataset, PATTERN_ENCODING

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Run Candlestick Pattern Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--step", type=str, default="all", choices=["all", "scrape", "download", "preprocess", "train_eval"], help="Specific step to run independently")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Running pipeline (step: {args.step}) with config: {config['pipeline_type']}")

    if args.step == 'scrape':
        run_scrape(config['data'])
        return
    elif args.step == 'download':
        run_download(config['data'])
        return
    elif args.step == 'preprocess':
        run_preprocess(config['data'])
        return
    elif args.step == 'train_eval':
        dataset = load_preprocessed_dataset(config['data'])
    else: # all
        dataset = load_and_preprocess_data(config['data'])
    
    if config['pipeline_type'] == 'two_stage':
        # Select classifier based on config
        if config['classifier']['name'] == 'minirocket':
            classifier = MiniRocketXGBClassifier(**config['classifier'].get('params', {}))
        else:
            classifier = MultiRocketXGBClassifier(**config['classifier'].get('params', {}))
            
        from src.localization.scanners import MultiWindowSlidingScanner
        from src.localization.clusterers import DBSCANClusterer
        
        scanner = MultiWindowSlidingScanner(
            window_sizes=[config['localization'].get('window_size', 30)], 
            stride=config['localization'].get('stride', 5),
            padding_proportion=config['localization'].get('padding_proportion', 0.2),
            probability_threshold=config['localization'].get('probability_threshold', 0.5),
            pattern_encoding_reversed={v: k for k, v in PATTERN_ENCODING.items()},
            n_jobs=1
        )
        clusterer = DBSCANClusterer(
            eps_base=config['localization'].get('eps_offset', 4)
        )
        localizer = Localizer(scanner=scanner, clusterer=clusterer)
        pipeline = TwoStagePipeline(classifier, localizer)
        
    result_dto = pipeline.run(dataset)
    
    evaluator = Evaluator()
    metrics = evaluator.evaluate(result_dto)
    
    logger.info("Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

if __name__ == "__main__":
    main()
