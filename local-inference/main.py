import argparse
import asyncio
from datetime import datetime

from utils import LogicManager, ProblemSetManager


async def main_loop(prompt_type, max_active, server_url, run_id, max_attempts, local_model_solves):
    problem_set_manager = ProblemSetManager()
    logic_manager = LogicManager(
        prompt_type=prompt_type,
        max_active=max_active,
        server_url=server_url,
        run_id=run_id,
        max_attempts=max_attempts,
        local_model_solves=local_model_solves,
    )
    queue_sem = asyncio.Semaphore(max_active + 1)
    tasks = set()

    def discard_and_ignore(task):
        tasks.discard(task)
        queue_sem.release()
        try:
            task.result()
        except Exception as e:
            print(f"Problem task failed: {repr(e)}")

    async def submit(request_type, request):
        return await logic_manager.handle(request_type, request)

    while (problem := problem_set_manager.get_next_problem()) is not None:
        await queue_sem.acquire()
        task = asyncio.create_task(submit("problem", problem))
        tasks.add(task)
        task.add_done_callback(discard_and_ignore)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(f"Completed run_id={run_id} with {len(results)} submitted problems.")


def build_parser():
    parser = argparse.ArgumentParser(description="Run local router over the problem set.")
    parser.add_argument(
        "--prompt-type",
        choices=["cache", "capabilities"],
        default="cache",
        help="Router prompt type.",
    )
    parser.add_argument(
        "--max-active",
        type=int,
        default=3,
        help="Maximum active router requests.",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8001",
        help="External resolution server base URL.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Pre-created run id. If omitted, a timestamped local id is generated.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum model attempts per problem before escalation/failure.",
    )
    parser.add_argument(
        "--local-model-solves",
        action="store_true",
        help="Allow local inference to solve very easy exact-match problems without an external model call.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Print router prompts and exit.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.test:
        for prompt_type in ("cache", "capabilities"):
            lm = LogicManager(
                prompt_type=prompt_type,
                max_active=args.max_active,
                server_url=args.server_url,
                run_id=args.run_id or "prompt_test",
                max_attempts=args.max_attempts,
                local_model_solves=args.local_model_solves,
            )
            print(f"\n--- {prompt_type} prompt ---\n")
            print(lm.router_prompt)
    else:
        run_id = args.run_id or f"{args.prompt_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        asyncio.run(
            main_loop(
                prompt_type=args.prompt_type,
                max_active=args.max_active,
                server_url=args.server_url,
                run_id=run_id,
                max_attempts=args.max_attempts,
                local_model_solves=args.local_model_solves,
            )
        )