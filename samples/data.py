from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd


# TRAIN A MODEL BASED ON OBSERVED SEARCHES
ARRAY_SIZE = 30

simulator = SearchSimulation()
training_data = simulator.observations(20, ARRAY_SIZE) #samples, array size

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
    # choose target location
    delta = model.predict(self.state())[0]
    new_location = self.smooth(self.location + delta)

    # check boundaries
    if new_location > self.ceil: new_location = self.ceil
    if new_location < self.floor: new_location = self.floor

    self.location = new_location



# BUILD TEST SAMPLES FOR EACH SEARCH TYPE

observation = simulator.observation(ARRAY_SIZE, supplied_search = AISearch)

print(observation)
