"""Offline inspect/validate/demo commands; deliberately no paid-call command."""
import argparse
import json
from .flow import ROOT, load, digest, questions, estimate
from .evaluation import protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["validate","questions","protocol","estimate"])
    parser.add_argument("--slider",choices=["calibration","constraint-precision","independent-judgment"],default="calibration")
    parser.add_argument("--input-rate",type=float)
    parser.add_argument("--output-rate",type=float)
    args=parser.parse_args()
    config,cases=load(args.slider)
    if args.command=="validate":
        result=dict(status="offline_ready_untrained_unjudged",slider=config["id"],cases=len(cases),
            config_sha256=digest(config),corpus_sha256=digest(cases),paid_calls=0,gpu_jobs=0)
    elif args.command=="questions":result=questions()
    elif args.command=="protocol":result=protocol(config)
    else:
        if args.input_rate is None or args.output_rate is None:
            parser.error("Estimate requires explicit rate inputs; this does not authorize calls")
        # Max counts across all three packs before payload/content deduplication.
        result=estimate([{"planned_request":i} for i in range(552)],input_usd_per_million=args.input_rate,
            output_usd_per_million=args.output_rate,input_token_bound=6000,output_token_bound=500,retry_reserve=56)
        result.update(selection_requests_max=168,heldout_scoring_requests_max=192,
            heldout_public_selection_requests_max=48,
            development_checkpoint_scoring="one checkpoint at alpha +1 included; further comparisons add up to 144 requests and generations each",
            development_scoring_requests_max=144,generation_calls_max=504,generation_output_token_cap=129024,
            input_bound_note="6000 is a planning input, not a verified tokenizer count; reserve must use provider billable-token bound")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
