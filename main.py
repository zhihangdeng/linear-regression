from src.master.master import Master
from src.slave.slave import Slave
from src.utils.config import Config


if __name__ == "__main__":
    config = Config()
    n = int(config.get("global", "n"))

    master = Master(config=config)
    slaves = [Slave(config=config, id=i) for i in range(n)]

    master.start()
    for slave in slaves:
        slave.start()

    master.join()
    for slave in slaves:
        slave.join()