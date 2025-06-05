from prefect import flow, task, get_run_logger

@task
def hello_task():
    logger = get_run_logger()
    logger.info("Hello, Prefect!")

@flow(name="hello-flow")
def my_flow():
    hello_task()

if __name__ == "__main__":
    my_flow()
