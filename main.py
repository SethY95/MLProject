"""
Python HE program
Based on the Zama library

#############################################################
#   This is a copy of the python code I've been dissecting  #
#   with help from Gemini and Copilot. AI has been used to  #
#   generate the base code for key gen, encryption,         #
#   decryption, and homomorphic addition. All notes and     #
#   documentation in this file were made by me and          #
#   represent my understanding and findings from analyzing  #
#   the generated code. The documentation provides useful   #
#   information like notes about the code structure, as     #
#   well as reasoning and logic behind the code. I will     #
#   continue to experiment with this program and update     #
#   my notes on a locally saved python file, meaning        #
#   updates wil not be reflected within this file.          #
#############################################################

Important Note!!
The subtraction function does not work 100% of the time. This
is because the usable space has not been configured to host
partitions of usable space to allow for negative numbers.
In order to successfully perform subtraction resulting with
negative numbers, the operational modulus space must be split
in half, with half representing a positive number and half
representing a negative number.

Info:   Homomorphic functions are notated as h_functionName.
Info:   The Modulus Q sets the size of the usable space or
        "total bit-budget" for encrypted data. See 0c for
        sizing information.
Info:   Q must be known to both the server and client to allow
        for computations to be done in the same space and ensure
        consistent overflow wrapping.
Info:   Encrypt creates ciphertexts as a tuple of two items:
        a 1D Numpy array and a single salar integer
Info:   The modulus Q does not allow a negative value, the
        max plaintext space must be halved and partitioned
        to account for negative value.
Info:   The multiplication function generates the product between
        an encrypted value and a non-encrypted value. For this
        reason, decrypt(ciphertext_B) is fed to the h_mult
        function as the plaintext.
"""
"""
Notes

0a......Learning with Errors is based on the concept of
        solving complex systems of linear equations with
        a small amount of intentional noise.
0b......Key vectors act like a lookup address to
        match a query vector (request) to
        relevant information (value vector).
0c......The Q Modulus must be large enough to contain space for the
        message layer (at the top bits), the guard layer (the middle
        buffer zone), and the noise layer (random errors at the bottom
        bits). If Q is too small, it will cause the data to overwrite
        itself and produces incorrect results.
1a......np.dot computes the dot product of two arrays.
1b......The dot product measures how much two vectors align to the same 
        direction.
"""
"""
Update Keypoints
->  The modulus was changed from 2**32 to 2**64 to provide enough
    space for homomorphic multiplication. Having the usable space
    as 2**32 resulted in large values overwriting vector data at
    the top of the matrix.
"""

import numpy as np

#Global Parameters (Toy setup)
Q = 2 ** 64  # Modulus (currently using 32-bit integers)

class Client:
    __N = 512  # Private key size
    __SCALE = 2 ** 20  # Scale factor to separate message from noise
    #__secret_key = None

    def __init__(self):
        self.generate_key()

    def generate_key(self, dim=__N):
        """
        Info: dim is the private key size
        Creates a random binary private key vector
        <--Note 0b-->
        """
        self.__secret_key = np.random.randint(0, 2, size=dim, dtype=np.uint32)

    def encrypt(self, plaintext, dim=__N, q=Q, scale=__SCALE):
        """
        Encrypts an integer message using LWE (learning with errors).
        <--Note 0a-->
        """

        # Generate a random matric/vector A
        a = np.random.randint(0, 2, size=dim, dtype=np.uint32)

        # Generate a small amount of noise
        error = int(np.random.normal(0, 2))

        # Compute b = (a * secret_key) + msg * scale + error (mod q)
        # Info: The message is shifted with SCALE to protect it from the noise
        # <--Note 1a and 1b-->
        b = (int(np.dot(a, self.__secret_key)) + (plaintext * self.__SCALE) + error) % q
        return a, b

    def decrypt(self, ciphertext, q=Q, scale=__SCALE):
        """
        Decrypt the LWE ciphertext
        """

        a, b = ciphertext

        # Compute b - (a * secret_key)
        # <--Note 1a and 1b-->
        raw = (b - int(np.dot(a, self.__secret_key))) % q

        # Divide by the scale and round to the nearest whole integer to strip noise
        return int(np.round(raw / self.__SCALE))


class Server:
    @staticmethod
    def h_add(ct1, ct2, q=Q):
        """Add two ciphertexts together"""
        a1, b1 = ct1
        a2, b2 = ct2

        # Linearly add the internal parts
        a_sum = (a1.astype(object) + a2.astype(object)) % q
        b_sum = (b1 + b2) % q

        return a_sum, b_sum

    @staticmethod
    def h_sub(ct1, ct2, q=Q):
        a1, b1 = ct1
        a2, b2 = ct2

        a_diff = (a1.astype(object) - a2.astype(object)) % q
        b_diff = (b1 - b2) % q

        return a_diff, b_diff

    @staticmethod
    def h_mult(ct, pt, q=Q):
        a, b = ct
        a_prod = (a.astype(object) * pt) % q
        b_prod = (b * pt) % q

        return a_prod, b_prod

class HomomorphicCrytography(Client, Server):
    pass

###########################################################################################################################
###########################################################################################################################

"""Federated Learning"""
class LocalNumPyClient:
    def __init__(self, client_id, x, y):
        self.client_id = client_id
        self.x = x #Private Local Features
        self.y = y #Private Local Targets
        self.n_samples = x.shape[0] #Returns the size of the first dimension of the multidimensional array or DataFrame
        self.weights = None
        self.bias = None

    def fit(self, global_weights, global_bias, lr=0.01, local_epochs=5):
        """Receives global parameters, trains locally, returns updates"""
        #Synchronize Local model parameters with the global server state
        self.weights = np.copy(global_weights)  #Creates a copy of global_weights
        self.bias = np.copy(global_bias)        #Creates a copy of global_bias

        #Local training Loop (Gradient Descent)
        for _ in range(local_epochs):
            """
            Prediction based on fundamental equation z = x * w + b
            self.x is the input data (features)
            self.weights is the learnable coefficients (parameters)
            self.bias is a learnable intercept that allows the activation function to shift
            np.dot performs matrix multiplication
            
            errors is made up of the difference between the prediction and target (from initialization)
            """
            predictions = np.dot(self.x, self.weights) + self.bias
            errors = predictions - self.y

            #Compute gradients for Linear Regression
            """
            The gradients determine how much to change each factor to minimize prediction error
            MSE (mean squared error) loss is used to calculate
            
            dw is the gradient for weights
            db is the gradient for bias
            """
            dw = (2 / self.n_samples) * np.dot(self.x.T, errors)
            db = (2 / self.n_samples) * np.sum(np.abs(errors))

            #Update Local parameters
            self.weights -= lr * dw
            self.bias -= lr * db

        #Returns updates parameters and Local dataset scale to the server
        return self.weights, self.bias, self.n_samples

    def evaluate(self, global_weights, global_bias):
        """Evaluates global model performance on private local validation data"""
        predictions = np.dot(self.x, global_weights) + global_bias

        #mse loss measures the average squared differences between errors (prediction - target)
        mse_loss = np.mean((predictions - self.y) ** 2)
        return mse_loss

#Central Server Logic (Replicating flwr.server.strategy.FedAvg)
class CentralFederatedServer:
    def __init__(self, input_dim):
        #Initialize the baseline global model parameters

        """
        global_weights is the global model weight assignments to each node
        """
        self.global_weights = np.zeros((input_dim, 1)) #Creates a 2D array filled with zeros with input_dim rows and 1 column
        self.global_bias = 0.0

    def aggregate(self, client_updates):
        """Computes the weighted Federating Averaging (FedAvg) math formula"""
        total_samples = sum(update[2] for update in client_updates)

        #Reset Global weights to accumulate the incoming client updates
        #Note: zeros_like creates an array with the same shape as the parameter
        new_weights = np.zeros_like(self.global_weights)
        new_bias = 0.0

        #Process every client package (weights, bias, sample count)
        for client_w, client_b, n_k in client_updates:
            weight_factor = n_k / total_samples
            new_weights += client_w * weight_factor
            new_bias += client_b * weight_factor

        #Update official global model state
        self.global_weights = new_weights
        self.global_bias = new_bias

###########################################################################################################################
###########################################################################################################################

class Message:
    def __init__(self, pt, cli):
        self.__pt = pt
        self.ct = None
        self.encrypt(cli)

    def getPT(self):
        return self.__pt

    def encrypt(self, cli):
        self.ct = cli.encrypt(self.__pt)

###########################################################################################################################
###########################################################################################################################
#####   Homomorphic Encryption Simulation   #####

he = HomomorphicCrytography()

#   --- Client Side: Key Gen and Encryption ---         #
client = Client()
server = Server()
#private_key = generate_key()
msg1 = Message(np.random.randint(100), client)
msg2 = Message(np.random.randint(100), client)
print(f"Msg1: {msg1.getPT()}, Encrypted A (Vector sample): {msg1.ct[0][:3]}..., Scalar: {msg1.ct[1]}")
print(f"Msg2: {msg2.getPT()}, Encrypted B (Vector sample): {msg2.ct[0][:3]}..., Scalar: {msg2.ct[1]}\n")

#   --- Server side: Zero-Knowledge Computation ---     #
encrypted_aresult = server.h_add(msg1.ct, msg2.ct)
encrypted_sresult = server.h_sub(msg1.ct, msg2.ct)
encrypted_mresult = server.h_mult(msg1.ct, client.decrypt(msg2.ct))
print(f"===Server Side===\nEncrypted Add Result\n{encrypted_aresult}\nEncrypted Mult Result\n{encrypted_mresult}\n")

#   --- Client side: Decryption ---                     #
decrypted_aresult = client.decrypt(encrypted_aresult)
decrypted_sresult = client.decrypt(encrypted_sresult)
decrypted_mresult = client.decrypt(encrypted_mresult)
print(f"---Result Decrypted by Client ---\n"
      f"Msg 1: {msg1.getPT()}\tMsg 2: {msg2.getPT()}\n"
      f"Result of homomorphic addition: {decrypted_aresult}\t(Expected: {msg1.getPT() + msg2.getPT()})\n"
      f"Result of homomorphic subtraction: {decrypted_sresult}\t(Expected: {msg1.getPT() - msg2.getPT()})\n"
      f"Result of homomorphic product: {decrypted_mresult}\t(Expected: {msg1.getPT() * msg2.getPT()})\n"
      )

###########################################################################################################################
###########################################################################################################################
#####   Federated Simulation Pipeline (Replicating flwr.simulation) #####

np.random.seed(42)
"""
Generate mock synthetic data partitioned across 3 distinct clients
Client 1: Small dataset (100 samples)
Client 2: Medium dataset (300 samples)
Client 3: Large dataset (600 samples)
"""
client_sizes = [100, 300, 600]
clients = []

#Theoretical target relationship we want global model to discover
true_w = np.array([[3.2], [-1.5]])  #Generates a multidimensional array of same-type data. This specifically creates a 2D column vector with floating-point numbers
true_b = 4.2

for idx, size in enumerate(client_sizes):
    """
        x_local is a 2D matrix of data points representing a single client's private data (independent variables)
        y_local is a 1D column vector containing "ground-truth answers" (dependent variables)
    """
    x_local = np.random.randn(size, 2)
    y_local = np.dot(x_local, true_w) + true_b * np.random.randn(size, 1) * 0.1
    clients.append(LocalNumPyClient(client_id=idx+1, x=x_local, y=y_local)) #Append and initialize a client

    #Initialize server orchestrator
    server = CentralFederatedServer(input_dim=2)
    federated_rounds = 4

    print("Beginning Federated Learning Iterations...\n" + ("-" * 50))

    for fl_round in range(1, federated_rounds + 1):
        round_client_updates = []

        #Broadcast and Local fit phase (Simulates network payload distribution)
        for client in clients:
            w_up, b_up, n_samples = client.fit(
                server.global_weights,
                server.global_bias,
                lr=0.05,
                local_epochs=3
            )
            #print(f"Up: {w_up}")
            round_client_updates.append((w_up, b_up, n_samples))

        #Global Aggregation phase on the server
        server.aggregate(round_client_updates)

        #Global Tracking evaluation metric phase
        round_losses = [client.evaluate(server.global_weights, server.global_bias) for client in clients]
        avg_round_loss = np.mean(round_losses)
        print(f"Round {fl_round} Complete | Global Average Validation MSE Loss: {avg_round_loss:.5f}")

    print("-" * 50)
    print("Federated Learning Complete!")
    print(f"Target True Weights:\n{true_w.flatten()} (Bias: {true_b})")
    print(f"Discovered Global Weights:\n{server.global_weights.flatten()} (Bias: {server.global_bias:.4f})")