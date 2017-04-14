import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# TRAIN A MODEL BASED ON OBSERVED SEARCHES
ARRAY_SIZE = 100
SAMPLE_SIZE = 100

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
    # choose target location
    delta = model.predict(self.state())[0]
    new_location = self.smooth(self.location + delta)

    # check boundaries
    if new_location >= self.ceil: new_location = self.ceil - 1
    if new_location < self.floor: new_location = self.floor

    self.location = new_location


# BUILD TEST SAMPLES FOR EACH SEARCH TYPE
rand = simulator.observations(SAMPLE_SIZE, ARRAY_SIZE,
  supplied_search = RandomSearch)

linear = simulator.observations(SAMPLE_SIZE, ARRAY_SIZE,
  supplied_search = LinearSearch)

binary = simulator.observations(SAMPLE_SIZE, ARRAY_SIZE,
  supplied_search = BinarySearch)

ai = simulator.observations(SAMPLE_SIZE, ARRAY_SIZE,
  supplied_search = AISearch)

print ("RANDOM: ", len(rand['deltas']) / SAMPLE_SIZE)
print ("LINEAR: ", len(linear['deltas']) / SAMPLE_SIZE)
print ("BINARY: ", len(binary['deltas']) / SAMPLE_SIZE)
print ("AI: ", len(ai['deltas']) / SAMPLE_SIZE)


# RANDOM:  97.89
# LINEAR:  30.61
# BINARY:  5.73
# AI:  3.24
