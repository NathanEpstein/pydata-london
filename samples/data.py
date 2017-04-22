import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from PyDexter import PyDexter

# TRAIN A MODEL BASED ON OBSERVED SEARCHES
ARRAY_SIZE = 100
SAMPLE_SIZE = 100
TRAINING_SIZE = 500

simulator = SearchSimulation()
training_data = simulator.observations(TRAINING_SIZE, ARRAY_SIZE)

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


rand_avg = len(rand['deltas']) / SAMPLE_SIZE
linear_avg = len(linear['deltas']) / SAMPLE_SIZE
binary_avg = len(binary['deltas']) / SAMPLE_SIZE
ai_avg = len(ai['deltas']) / SAMPLE_SIZE

pydex = PyDexter()
pydex.bar({
  'labels': ["Random", "Linear", "Binary", "Forest"],
  'groups': [""],
  'datasets': [
    {
      'values': [rand_avg],
    },
    {
      'values': [linear_avg],
    },
    {
      'values': [binary_avg],
    },
    {
      'values': [ai_avg]
    }
  ]
})

# RANDOM:  98.34
# LINEAR:  31.5
# BINARY:  5.87
# AI:  3.32
