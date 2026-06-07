import asyncio

from utils import ProblemSetManager, LogicManager

async def main_loop(prompt_type, max_active):
    problem_set_manager = ProblemSetManager()
    logic_manager = LogicManager(prompt_type, max_active)
    queue_sem = asyncio.Semaphore(max_active + 1)
    tasks = set()

    def discard_and_ignore(task):
        tasks.discard(task)
        queue_sem.release()
        try:
            task.result()
        except Exception:
            pass
    
    async def submit(request_type, request):
        return await logic_manager.handle(request_type, request)
        
    while (problem := problem_set_manager.get_next_problem()) is not None:
        await queue_sem.acquire()
        task = asyncio.create_task(submit("problem", problem))
        tasks.add(task)
        task.add_done_callback(discard_and_ignore)
        
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        for prompt_type in ("cache", "capabilities"):
            lm = LogicManager(prompt_type, max_active=4)
            print(f"\n--- {prompt_type} prompt ---\n")
            print(lm.router_prompt)
    else:
        asyncio.run(main_loop("cache", max_active=3))