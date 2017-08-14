from multiprocessing import Process

class Scheduler:
	def __init__(self):		
		self.processes = []

	def schedule(self, function, args):
		process = Process(target = function, args = args)
		self.processes.append(process)
		process.start()

	def joinAll(self):
		for process in self.processes:
			process.join()

