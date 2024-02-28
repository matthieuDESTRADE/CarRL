### RL PROJECT

This project aims to implement a reinforcement learning algorithm to create an autonomous agent able to play to a simple racing game.

async training using replay buffer filled with experiences



https://web.stanford.edu/class/aa228/reports/2018/final150.pdf
=> implement a transfer learning from pre-trained CNN

what about genetic algorithms
https://www.youtube.com/watch?v=SX08NT55YhA

training:
spawn not only at random angle and position but also at random speed

example of model to use for deep q network:
https://arxiv.org/pdf/1807.02371.pdf
=> this one is used for a more complex game (high graphics and unprevisibility + more advanced dynamics: for steering [-1, 1], gas [0, 1], brake {0, 1} and hand brake {0, 1}. Note that the brake and hand brake commands are binary)

idea from playing atari games:
https://arxiv.org/pdf/1312.5602.pdf
 we also use a simple frame-skipping technique [3]. More precisely, the agent sees and selects actions on every k
th frame instead of every
frame, and its last action is repeated on skipped frames. Since running the emulator forward for one
step requires much less computation than having the agent select an action, this technique allows
the agent to play roughly k times more games without significantly increasing the runtime.

model used:
The input to the neural network consists is an 84 × 84 × 4 image produced by φ. The first hidden layer convolves 16 8 × 8
filters with stride 4 with the input image and applies a rectifier nonlinearity [10, 18]. The second
hidden layer convolves 32 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity. The
final hidden layer is fully-connected and consists of 256 rectifier units. The output layer is a fullyconnected linear layer with a single output for each valid action


https://www.nature.com/articles/s41586-021-04357-7#Sec5
GT racing env => complex reward :
 course progress(Rcp), off-course penalty (Rsoc or Rloc), wall penalty (Rw), tyre-slip penalty (Rts), passing bonus (Rps), any-collision penalty (Rc), rear-end penalty (Rr) and unsporting-collision penalty (Ruc)