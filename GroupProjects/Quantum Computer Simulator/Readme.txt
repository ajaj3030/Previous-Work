README

Attached in this folder is the quantum computer simulator. This was a group project with primarily 3 people working on the code. My contributions were mainly concentrated in the GUI/Testing and developing of the algorithms (so basically not as much on the initial quantum gates) for code implementation. I contributed towards the theoretical working of all components.

Grover is a search algorithm. We designate our target state and use a series of quantum gates to amplify the probability of measuring the target state as can be seen in the testing. I would recommend reading through the qiskit site if more background knowledge is desired.

Shor is a factorisiation algorithm. We feed in a number and the algorithm should return the two largest prime factors. It does this by exploiting the largest common devisor, the inverse quantum fourier transform and Fermat's little theroem. This is much more complicated, and the algorithm has limited success factoring numbers that aren't small. This is normal; we are limited by the number of qubits in the simulator. Increasing the number of qubits would require the development of more advanced control gates, and given this was an optional part of the project already we decided not to push too far.
