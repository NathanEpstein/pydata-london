from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd


# TRAIN A MODEL BASED ON OBSERVED SEARCHES
ARRAY_SIZE = 100

simulator = SearchSimulation()
training_data = simulator.observations(500, ARRAY_SIZE) #samples, array size

states = pd.DataFrame(training_data['states'])
deltas = pd.DataFrame(training_data['deltas'])

model = RandomForestRegressor()
model = model.fit(states, deltas.values.ravel())

class AISearch(Search):
  def smooth(self, num):
    return random.choice([
      math.floor,
      math.ceil
    ])(num)

  def update_location(self):

    delta = model.predict(self.state())[0]
    new_location = int(round(self.location + delta))

    # check boundaries
    if new_location > self.ceil: new_location = self.ceil
    if new_location < self.floor: new_location = self.floor

    old_location = self.location
    self.location = new_location

    print({
      'prev_location': old_location,
      'delta': delta,
      'new_location': self.location
      })


array = simulator._random_sorted_array(ARRAY_SIZE)
search = AISearch(array, random.choice(array))
observation = simulator.observation(ARRAY_SIZE, supplied_search = search)

# print(observation)
