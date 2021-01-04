import argparse
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from simulator import Simulator, WORLD_WIDTH, WORLD_HEIGHT
from racetrack import RaceTrack, Contour, Horizontals, load_racetrack
import datetime

LAP_TRIM_IDX = 690

def convert(x,y):
    new_x = np.array(x[:LAP_TRIM_IDX])
    new_y = 800 - np.array(y[:LAP_TRIM_IDX])
    return new_x, new_y

def main():
    
    #added for time efficiency measurement
    begin_time = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--which", help="which filter to run (write pf for particle filter, kf for Kalman filter)")
    parser.add_argument("-n", "--num_particles", default=50, type=int, help='Number of particles for particle filtering')
    parser.add_argument("-m", "--max_sensor_range", default=50, type=int, help='Maximum range of the car\'s sensors')
    parser.add_argument("-s", "--sensor_noise_std", default=0.0, type=float, help='Std dev of car\'s sensor noise')
    parser.add_argument("-d", "--gps_noise_dist", default="gaussian", help='Type of distribution for GPS sensor noise (gaussian or uniform)')
    parser.add_argument("-gv", "--gps_noise_var", default=10.0, type=float, help='Variance of gaussian noise for GPS measurement (Kalman filter)')
    parser.add_argument("-gw", "--gps_noise_width", default=20, type=float, help='Width of uniformly random noise for GPS measurement (Kalman filter)')
    parser.add_argument("-f", "--filename", default="plot.png", help="name of image to store plot inside the plots/ directory")
    args = parser.parse_args()

    car_pos_x = []
    car_pos_y = []
    est_pos_x = []
    est_pos_y = []
    gps_x = []
    gps_y = []

    if args.which == "pf":
        max_sensor_range = args.max_sensor_range
        sensor_std = args.sensor_noise_std
        num_particles = args.num_particles
        print("Running particle filtering with\n    Num particles = {}\n    Max sensor range = {}\n    Sensor noise std = {}".format(num_particles, max_sensor_range, sensor_std))
        sim = Simulator(max_sensor_range=max_sensor_range, sensor_std=sensor_std, num_particles=num_particles)
        sim.toggle_particles()
    elif args.which == "kf":
        gps_noise_dist = args.gps_noise_dist
        if gps_noise_dist == "gaussian":
            gps_noise_var = args.gps_noise_var
            sim = Simulator(gps_noise_var=gps_noise_var)
            sim.gps_noise_dist = gps_noise_dist
            print("Running Kalman filtering with\n    GPS noise dist = {}\n    GPS gaussian noise var = {}".format(gps_noise_dist, gps_noise_var))
        elif gps_noise_dist == "uniform":
            gps_noise_width = args.gps_noise_width
            sim = Simulator(gps_noise_width=gps_noise_width)
            sim.gps_noise_dist = gps_noise_dist
            print("Running Kalman filtering with\n    GPS noise dist = {}\n    GPS uniform noise width = {}".format(gps_noise_dist, gps_noise_width))
        else:
            raise ValueError
        sim.toggle_kalman()
    else:
        raise ValueError

    if args.which in ["pf","kf"]:
        sim.toggle_replay()

        segment = 0
        display_progress_every = 25

        for i in range(LAP_TRIM_IDX):
            sim.loop()
            car_pos_x.append(sim.car.pos[0])
            car_pos_y.append(sim.car.pos[1])
            if args.which == "pf":
                est_pos_x.append(sim.x_est)
                est_pos_y.append(sim.y_est)
            elif args.which == "kf":
                est_pos_x.append(sim.kf_state[0])
                est_pos_y.append(sim.kf_state[1])
                gps_x.append(sim.gps_measurement[0])
                gps_y.append(sim.gps_measurement[1])
            if int(i % ((LAP_TRIM_IDX - 1) / (100 / display_progress_every))) == 0:
                print("{}% complete".format(segment * display_progress_every))
                segment += 1

        car_pos_x, car_pos_y = convert(car_pos_x, car_pos_y)
        est_pos_x, est_pos_y = convert(est_pos_x, est_pos_y)
        if args.which == "kf":
            gps_x, gps_y = convert(gps_x, gps_y)
        error = []
        for i in range(len(car_pos_x)):
            car_pos = np.array([car_pos_x[i],car_pos_y[i]])
            pf_pos = np.array([est_pos_x[i],est_pos_y[i]])
            error.append(np.linalg.norm(car_pos - pf_pos))

        #added for time efficiency measurement
        end_time = datetime.datetime.now() - begin_time

        fig, axs = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(10, 10)

        if args.which == "pf":
            fig.suptitle("Particle Filtering, N={}, Range={}, Noise={}".format(sim.num_particles, sim.max_sensor_range, sim.sensor_std))
        elif args.which == "kf":
            if args.gps_noise_dist == "gaussian":
                fig.suptitle("Kalman Filtering, GPS Noise Dist = Gaussian, GPS Noise Var = {}".format(sim.gps_noise_var))
            elif args.gps_noise_dist == "uniform":
                fig.suptitle("Kalman Filtering, GPS Noise Dist = Uniform, GPS Noise Width = {}".format(sim.gps_noise_width))

        ax0 = axs[0]
        ax1 = axs[1]

        ax0.title.set_text("Path of estimated position")
        ax0.plot(car_pos_x, car_pos_y, label="car position", color="black")
        ax0.plot(est_pos_x, est_pos_y, label="estimated position", color="blue")
        if args.which == "kf":
            ax0.plot(gps_x, gps_y, label="GPS measurement", color="red")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.set_xlim(0,WORLD_WIDTH)
        ax0.set_ylim(0,WORLD_HEIGHT)
        ax0.set_xticks(np.arange(0, WORLD_WIDTH, 50))
        ax0.set_xticks(np.arange(0, WORLD_WIDTH, 10), minor=True)
        ax0.set_yticks(np.arange(0, WORLD_HEIGHT, 50))
        ax0.set_yticks(np.arange(0, WORLD_HEIGHT, 10), minor=True)
        ax0.grid(which='minor', alpha=0.2)
        ax0.grid(which='major', alpha=0.5)
        ax0.legend()

        ax1.title.set_text("Error in estimated position (Euclidean distance)")
        ax1.plot(error)
        #added to Time label for time efficiency measurement
        ax1.set_xlabel("Time: {}".format(end_time))
        ax1.set_ylabel("Error")
        ax1.set_xlim(0,LAP_TRIM_IDX)
        ax1.set_xticks(np.arange(0,LAP_TRIM_IDX,50))
        ax1.set_xticks(np.arange(0,LAP_TRIM_IDX,10), minor=True)
        ax1.set_yticks(np.arange(0,np.max(error),100))
        ax1.set_yticks(np.arange(0,np.max(error),20), minor=True)
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.5)

        plt.savefig("plots/"+args.filename, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
    
    
