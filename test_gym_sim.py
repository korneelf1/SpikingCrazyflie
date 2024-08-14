import unittest
from gym_sim import *
from libs.cpuKernels import controller, step
sim_N1 = Drone_Sim(drone='og', N_drones=1)

sim_N3 = Drone_Sim(drone='og', N_drones=3)     
class TestGymSim(unittest.TestCase):
    def test_check_done(self):
        '''
        Test the done condition for:
        - position
        - velocity
        - angular velocity
        - time

        for both sim with 1 drone and 3 drones
        '''
        # done condition 1: abs(position) greater than 0.6
        x1 = np.zeros((1,17)).astype(np.float32)
        x1[:,:3] = 0.7

        t1 = np.ones((1,)).astype(np.float32) 
        sim_N1.xs = x1
        sim_N1.t = t1

        sim_N1._check_done()
        done = sim_N1.done.sum() # should be 1
        assert done == 1

        x3 = np.zeros((3,17)).astype(np.float32)
        x3[1:,:3] = 0.7 # only the second and third drones are out of bounds

        t3 = np.ones((3,)).astype(np.float32) 
        print(sim_N3.t.shape)
        sim_N3.xs = x3
        sim_N3.t = t3

        sim_N3._check_done()
        done = sim_N3.done.sum()

        assert done == 2
        sim_N1.reset()
        sim_N3.reset()
        # done condition 2: abs(velocity) greater than 1e3
        x1 = np.zeros((1,17)).astype(np.float32)
        x1[:,3:6] = 1e4

        t1 = np.ones((1,)).astype(np.float32) * 0.1
        sim_N1.xs = x1
        sim_N1.t = t1

        sim_N1._check_done()
        done = sim_N1.done.sum()

        assert done == 1

        x3 = np.zeros((3,17)).astype(np.float32)
        x3[:2,3:6] = 1e4

        t1 = np.ones((3,)).astype(np.float32) * 0.1
        sim_N3.xs = x3
        sim_N3.t = t1

        sim_N3._check_done()
        done = sim_N3.done.sum()

        assert done == 2
        # done condition 3: abs(angular velocity) greater than 1e3
        sim_N1.reset()
        sim_N3.reset()

        x1 = np.zeros((1,17)).astype(np.float32)
        x1[:,10:13] = 1e4

        t1 = np.ones((1,)).astype(np.float32) * 0.1
        sim_N1.xs = x1
        sim_N1.t = t1

        sim_N1._check_done()
        done = sim_N1.done.sum()

        assert done == 1

        x3 = np.zeros((3,17)).astype(np.float32)
        x3[2:,10:13] = 1e4

        t1 = np.ones((3,)).astype(np.float32) * 0.1
        sim_N3.xs = x3
        sim_N3.t = t1

        sim_N3._check_done()
        done = sim_N3.done.sum()

        assert done == 1
        # done condition 4: time greater than 5 seconds
        sim_N1.reset()
        sim_N3.reset()

        x1 = np.zeros((1,17)).astype(np.float32)

        t1 = np.ones((1,)).astype(np.float32) * 601
        sim_N1.xs = x1
        sim_N1.t = t1

        sim_N1._check_done()
        done = sim_N1.done.sum()
        print(done)
        assert done == 1

        x3 = np.zeros((3,17)).astype(np.float32)

        t1 = np.ones((3,)).astype(np.float32) * 1e3
        sim_N3.xs = x3
        sim_N3.t = t1

        sim_N3._check_done()
        done = sim_N3.done.sum()

        assert done == 3

    def test_step_rollout_equivalence(self):
        # Test if step_rollout will produce the same result as just the original step kernel
        # for both sim with 1 drone and 3 drones
        x1, info1 = sim_N1.reset()
        x3, info3 = sim_N3.reset()

        us1 = np.random.random((1, 4)).astype(np.float32)
        us3 = np.random.random((3, 4)).astype(np.float32)

        # for i in range(100):
        #     x1, info1 = sim_N1.step(us1)
        #     x3, info3 = sim_N3.step(us3)
        pass

    def test_step_equivalence(self):
        # Test if step of original sim and wrapped with gym are equivalent
        xs_fpdsim = np.array([[[ 1.4393571e-01, -3.9724422e-01,1.9863263e-01,3.9972389e-01,
                            3.9143974e-01,1.8604547e-01,-2.2311960e-01,3.2185945e-01,
                            7.9460442e-01,4.6392673e-01,2.1806990e-01,-3.0284607e-01,
                            6.8650335e-01,1.1996868e+03,1.0236104e+03,1.2446268e+03,
                            6.5556470e+02]],
                            [[ 1.4793295e-01,-3.9332983e-01,2.0049308e-01,4.0121105e-01,
                            3.6802298e-01,2.9660529e-01,-2.2385804e-01,3.2504368e-01,
                            7.9433727e-01,4.6180359e-01,9.1028467e-02,-6.8054634e-01,
                            5.2868104e-01,1.2619869e+03,1.4087151e+03,8.2975122e+02,
                            7.3951239e+02]],
                            [[ 1.5194505e-01,-3.8964960e-01,2.0345913e-01,4.0279874e-01,
                            3.4283829e-01,4.0822831e-01,-2.2252171e-01,3.2860985e-01,
                            7.9444247e-01,4.5973995e-01,-4.9440247e-01,-6.1798167e-01,
                            5.4585469e-01,1.2445692e+03,1.6942178e+03,5.5316748e+02,
                            7.6201221e+02]],
                            [[ 1.5597305e-01,-3.8622123e-01,2.0754141e-01,4.0442246e-01,
                            3.1514865e-01,5.2139938e-01,-2.2050683e-01,3.3274490e-01,
                            7.9308754e-01,4.6007580e-01,-1.3775698e+00,-2.1895930e-01,
                            5.4561388e-01,1.2205503e+03,1.8739211e+03,3.6877832e+02,
                            8.0775488e+02]],
                            [[ 1.6001727e-01,-3.8306975e-01,2.1275540e-01,4.0592459e-01,
                            2.8494161e-01,6.3598794e-01,-2.1859565e-01,3.3692154e-01,
                            7.8923017e-01,4.6455958e-01,-2.4307842e+00,4.1561773e-01,
                            5.4334080e-01,1.1893385e+03,1.9748960e+03,2.4585220e+02,
                            8.5692267e+02]],
                            [[1.6407651e-01,-3.8022032e-01,2.1911527e-01,4.0708846e-01,
                            2.5291809e-01,7.5124836e-01,-2.1738558e-01,3.4072992e-01,
                            7.8215206e-01,4.7422037e-01,-3.5607793e+00,1.2055080e+00,
                            5.3432047e-01,1.1553315e+03,2.0147305e+03,1.6390146e+02,
                            9.0312909e+02]],
                            [[ 1.6814739e-01,-3.7769115e-01,2.2662775e-01,4.0771562e-01,
                            2.1988116e-01,8.6623186e-01,-2.1726148e-01,3.4376949e-01,
                            7.7134943e-01,4.8953047e-01,-4.6986351e+00,2.0837896e+00,
                            5.1750821e-01,1.1198641e+03,2.0071011e+03,1.0926764e+02,
                            9.4153479e+02]],
                            [[ 1.7222454e-01,-3.7549233e-01,2.3529007e-01,4.0766403e-01,
                            1.8660051e-01,9.7998315e-01,-2.1841572e-01,3.4565386e-01,
                            7.5644332e-01,5.1050115e-01,-5.7946658e+00,2.9960165e+00,
                            4.9262652e-01,1.0828063e+03,1.9622277e+03,7.2845093e+01,
                            9.6967365e+02]],
                            [[ 1.7630118e-01,-3.7362632e-01,2.4508990e-01,4.0686557e-01,
                            1.5379848e-01,1.0916692e+00,-2.2087188e-01,3.4601292e-01,
                            7.3713493e-01,5.3677070e-01,-6.8139744e+00,3.8992236e+00,
                            4.6004522e-01,1.0430293e+03,1.8876514e+03,4.8563393e+01,
                            9.8674658e+02]],
                            [[ 1.8036984e-01,-3.7208834e-01,2.5600660e-01,4.0533033e-01,
                            1.2217267e-01,1.2006613e+00,-2.2451575e-01,3.4450245e-01,
                            7.1319407e-01,5.6768399e-01,-7.7327518e+00,4.7608805e+00,
                            4.2045638e-01,9.9889240e+02,1.7887520e+03,3.2375595e+01,
                            9.9298523e+02]]])
        x1, info1 = sim_N1.reset()
        x3, info3 = sim_N3.reset()

        itaus1 = sim_N1.itaus
        omegaMaxs1 = sim_N1.omegaMaxs
        G1s1 = sim_N1.G1s
        G2s1 = sim_N1.G2s
        dt1 = sim_N1.dt
        xs_log1 = sim_N1.xs_log
        G1pinvs1 = np.linalg.pinv(sim_N1.G1s) / (omegaMaxs1*omegaMaxs1)[:, :, np.newaxis]

        itaus3 = sim_N3.itaus
        omegaMaxs3 = sim_N3.omegaMaxs
        G1s3 = sim_N3.G1s
        G2s3 = sim_N3.G2s
        dt3 = sim_N3.dt
        xs_log3 = sim_N3.xs_log
        G1pinvs3 = np.linalg.pinv(sim_N3.G1s) / (omegaMaxs3*omegaMaxs3)[:, :, np.newaxis]

        log_idx1 = int(1)
        log_idx3 = int(1)

        us1 = np.random.random((1, 4)).astype(np.float32)
        us3 = np.random.random((3, 4)).astype(np.float32)

        sim_N1.xs = x1.copy()

        sim_N3.xs = x3.copy()

        posPs1 = 2*np.ones((1, 3), dtype=np.float32)
        velPs1 = 2*np.ones((1, 3), dtype=np.float32)

        posPs3 = 2*np.ones((3, 3), dtype=np.float32)
        velPs3 = 2*np.ones((3, 3), dtype=np.float32)
        for i in range(100):
            step(x1, us1, itaus1, omegaMaxs1, G1s1, G2s1, dt1, log_idx1, xs_log1)
            step(x3, us3, itaus3, omegaMaxs3, G1s3, G2s3, dt3, log_idx3, xs_log3)

            gymsimx1 = sim_N1.step(us1)[0]
            gymsimx3 = sim_N3.step(us3)[0]

            assert np.allclose(x1[:,:17],gymsimx1[:,:17])
            assert np.allclose(x3[:,:17], gymsimx3[:,:17])

            controller(x1, us1, posPs1,velPs1, sim_N1.pSets, G1pinvs1)
            controller(x3, us3, posPs3,velPs3, sim_N3.pSets, G1pinvs3)
    
        pass

if __name__ == '__main__':
    unittest.main()