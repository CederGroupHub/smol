def run(self, iterations, sublattices=None):
    """Run the ensemble for the given number of iterations.

    Samples are taken at the set intervals specified in constructur.

    Args:
        iterations (int):
            Total number of monte carlo steps to attempt
        sublattices (list of str):
            List of sublattice names to consider in site flips.
    """
    write_loops = iterations // self.sample_interval
    if iterations % self.sample_interval > 0:
        write_loops += 1

    start_step = self.current_step

    for _ in range(write_loops):
        remaining = iterations - self.current_step + start_step
        no_interrupt = min(remaining, self.sample_interval)

        for _ in range(no_interrupt):
            success = self._attempt_step(sublattices)
            self._ssteps += success

        self._step += no_interrupt
        self._save_data()


    def dump(self, filename):
        """Write data into a text file in json format, and clear data."""
        with open(filename, 'a') as fp:
            for d in self.data:
                json.dump(d, fp)
                fp.write(os.linesep)
        self._data = []

    def _get_current_data(self):
        """Extract the ensemble data from the current state.

        Returns: ensemble data
            dict
        """
        return {'occupancy': self.current_occupancy}

    def _save_data(self):
        """
        Save the current sample and properties.

        Args:
            step (int):
                Current montecarlo step
        """
        data = self._get_current_data()
        data['step'] = self.current_step
        self._data.append(data)

    @abstractmethod
    def _attempt_step(self, sublattices=None):
        """Attempt a MC step and return if the step was accepted or not."""
        return




"""
initial_occupancy (ndarray or list):
    Initial occupancy vector. The occupancy can be encoded
    according to the processor or the species names directly.
"""

if len(initial_occupancy) != len(processor.structure):
    raise ValueError('The given initial occupancy does not match '
                     'the underlying processor size')

if isinstance(initial_occupancy[0], str):
    initial_occupancy = processor.encode_occupancy(initial_occupancy)