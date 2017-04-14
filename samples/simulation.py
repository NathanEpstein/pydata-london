# simulation class which uses searches to create training data
class SearchSimulation:
  def observation(self, array_length, supplied_search = None):
    if supplied_search is None:
      search = self._search_of_random_type(array_length)
    else:
      search = supplied_search

    # location of the target value
    target_location = search.array.index(search.target)
    # observed states and proscribed transitions
    states = []
    deltas = []

    while (search.location != target_location):
      state = search.state()
      delta = [target_location - search.location]

      states.append(state)
      deltas.append(delta)

      search.update()

    return { 'states': states, 'deltas': deltas }

  def observations(self, n, array_length):
    states = []
    deltas = []

    for i in range(n):
      observation = self.observation(array_length)
      states += observation['states']
      deltas += observation['deltas']

    return { 'states': states, 'deltas': deltas }

  def _search_of_random_type(self, array_length):
    sorted_array = self._random_sorted_array(array_length)
    target_int = random.choice(sorted_array)

    search_type = random.choice(['binary', 'linear', 'random'])
    if search_type == 'binary':
      search = BinarySearch(sorted_array, target_int)
    if search_type == 'random':
      search = RandomSearch(sorted_array, target_int)
    if search_type == 'linear':
      search = LinearSearch(sorted_array, target_int)

    return search

  def _random_sorted_array(self, length):
    random_values = []
    for i in range(length):
      value = round(random.expovariate(1), 5)
      random_values.append(value)

    random_values.sort()
    return random_values

