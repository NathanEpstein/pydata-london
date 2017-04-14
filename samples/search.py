import random
import math

class Search:
  def __init__(self, array, target):
    self.path = []
    self.array = array
    self.target = target
    self.initialize_features()

  def initialize_features(self):
    self.location = random.choice(range(len(self.array)))
    self.floor = 0
    self.ceil = len(self.array)
    self.ratio = self.current() / self.target

  def current(self):
    return self.array[self.location]

  def state(self):
    features = [
      self.location,
      self.floor,
      self.ceil,
      self.ratio
    ]
    return features

  def update(self):
    self.update_location() #supplied by child class
    self.path.append(self.location)

    #update ratio
    self.ratio = self.current() / self.target
    #update floor
    if (self.ratio < 1): self.floor = self.location
    #update ceil
    if (self.ratio > 1): self.ceil = self.location

    return self.location


# specific search classes inherit from Search and supply update_location method
class BinarySearch(Search):
  def update_location(self):
    self.location = math.floor((self.ceil + self.floor) / 2)

class LinearSearch(Search):
  def update_location(self):
    if self.ratio < 1:
      self.location += 1
    else:
      self.location -= 1

class RandomSearch(Search):
  def update_location(self):
    self.location = random.choice(range(len(self.array)))

