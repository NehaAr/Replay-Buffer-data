import random
import psutil

def calculate_available_memory():
  available_memory=psutil.virtual_memory().available
  ram_available=available_memory/(1024**2)
  return ram_available

class Replay_Buffer:

   def __init__(self,batch_size):
       self.capacity=int((calculate_available_memory()*0.2))
       self.buffer=[]
       self.index=0
       self.batch_size=batch_size
       self.reservoir=[]

   def add(self,sample):                                    
     if len(self.buffer)<=self.capacity:
        self.buffer.append(sample)
     else:
        self.buffer[self.index]=sample
     self.index=(self.index+1)%self.capacity

   def sample_data(self,batch_size):                           
      return random.sample(self.buffer, batch_size)

   def reservoir_sampling(self,batch_size):                   
      for i,sample in enumerate(self.buffer):
        if i<=self.batch_size:
          self.reservoir.append(sample)
        else:
          random_prob=random.random()
          if random_prob<self.batch_size/i:
             replace_index=random.randint(0,self.batch_size-1)
             self.reservoir[replace_index]=sample

   def size(self):
        """
        Get the current size of the buffer.
        :return: The current number of samples in the buffer.
        """
        return len(self.buffer)

   def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.index = 0
