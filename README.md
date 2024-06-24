# PiKNODE
**Title**

Forecasting dynamics by an incomplete equation of motion and an auto-encoder Koopman operator

**Authors**  
Zhao Chen (Southeast University)  
Hao Sun (Renmin University of China)
Wen Xiong (Southeast University)

**Abstract**  
The governing equation of mass–spring–damper dynamics usually consists of a known structure and undetermined nonlinear components. This work proposes a physics-informed method to learn the unknown component by modeling its evolution with an auto-encoder Koopman operator. The Koopman operator is well known for its global linearization of dynamics and sensitivity to spectral modes. The undetermined component in an oscillatory equation of motion (EOM) is first reconstructed by a time-delay neural network whose inputs stack past sequences of observables and the control. Then, the evolution of the undetermined component is globally linearized and propagated by the Koopman operator in a latent space created by an auto-encoder neural network. Ultimately, observables at the future timestep are forecasted by timestepping the original EOM. Uniting an incomplete EOM and the Koopman operator, the proposed physics-informed Koopman neural ordinary differential equation (PiKNODE) method shows competitive edges compared with several black- and grey-box benchmarks in four synthetic and experimental case studies. Additionally, an anomaly-detection technique based on the Jensen–Shannon distance is proposed to boost the practicality of the PiKNODE for structural health monitoring.

**Graphical Abstract**  
<img src="concept.jpg" alt="drawing" width="750"/>

**Journal**  
Mechanical Systems and Signal Processing

**Paper Link**  
https://www.sciencedirect.com/science/article/pii/S0888327024004977#d1e2174
