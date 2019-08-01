import random

import numpy as np

import carla
import time

from constants import *


# Random spawn location.
# These numbers are valid for only a particular junction scenario.
def random_location(rand):
    # Spawn on left side
    if rand < 0.5:
        return carla.Transform(carla.Location(x=random.randint(20, 60), y=random.choice([130.5, 127.5]), z=0.2),
                               carla.Rotation(yaw=180))
    # Spawn on right side
    else:
        return carla.Transform(carla.Location(x=-random.randint(20, 60), y=random.choice([134.5, 137.5]), z=0.2),
                               carla.Rotation(yaw=0))


class Simulator(object):
    # Simulator state
    collided = False
    frames = []

    def __init__(self, args):
        # initialise Carla client
        self.world = carla.Client(args.host, args.port).get_world()
        self.bplib = self.world.get_blueprint_library()
        self.n_vehicles = args.n_vehicles

        self.autocar = None
        self.av_sensor = None
        self.collision_sensor = None

        self.vehicles = []
        self.actors = []

    def reset(self):
        self.destroy()

        # Spawn Autonomous car.
        # Spawn position is random in a certain range. Numbers valid for only particular junction.
        self.autocar = self.world.spawn_actor(self.bplib.find('vehicle.tesla.model3'),
                                              carla.Transform(carla.Location(5, random.randint(100, 120), 0.2), carla.Rotation(yaw=90)))
        # Apply brake as default control
        self.autocar.apply_control(carla.VehicleControl(brake=1))
        self.actors.append(self.autocar)

        # Spawn Camera
        # Choose between rgb and semantic segmentation
        # camera_properties = self.bplib.find('sensor.camera.rgb')
        camera_properties = self.bplib.find('sensor.camera.semantic_segmentation')
        camera_properties.set_attribute('fov', '110')
        camera_properties.set_attribute('image_size_x', str(IMG_WIDTH))
        camera_properties.set_attribute('image_size_y', str(IMG_HEIGHT))
        camera_properties.set_attribute('sensor_tick', str(1 / FREQUENCY))
        self.av_sensor = self.world.spawn_actor(camera_properties,
                                                carla.Transform(carla.Location(0, 132, 40), carla.Rotation(roll=90, pitch=-90)),
                                                attach_to=self.autocar
                                                )
        self.av_sensor.listen(lambda image: Simulator.new_frame(image))
        self.actors.append(self.av_sensor)

        # Choose a random number of NPC to spawn
        vehicles_to_spawn = random.randint(1, self.n_vehicles)
        tries = 0
        while vehicles_to_spawn > 0 or tries < vehicles_to_spawn * 10:

            # Spawn NPC
            vehicle = self.world.try_spawn_actor(self.bplib.find('vehicle.audi.a2'),
                                                 random_location(np.random.rand()))

            # if spawning failed, retry
            if vehicle is None:
                continue

            self.vehicles.append(vehicle)
            self.actors.append(vehicle)
            vehicles_to_spawn -= 1
            tries += 1

        # Spawn Collision detector.
        self.collision_sensor = self.world.spawn_actor(self.bplib.find('sensor.other.collision'),
                                                       carla.Transform(),
                                                       attach_to=self.autocar)
        self.collision_sensor.listen(lambda x: Simulator.on_collision())
        self.actors.append(self.collision_sensor)
        Simulator.collided = False

        # Wait for the simulator to get ready
        time.sleep(1 / FREQUENCY)

        return Simulator.state()

    def step(self, action):

        # apply default control to NPCs
        for vehicle in self.vehicles:
            vehicle.apply_control(carla.VehicleControl(throttle=0.5))

        # Note old AV location to calculate reward
        old_y = self.autocar.get_location().y

        # apply action to AV
        self.autocar.apply_control(carla.VehicleControl(throttle=THROTTLE[action], brake=BRAKE[action]))

        # sleep to observe action's effect
        time.sleep(1 / FREQUENCY)

        # calculate reward (\Delta s + collision penalty)
        reward = (self.autocar.get_location().y - old_y) + (-100 if Simulator.collided else 0)

        return {'state': self.state(),
                'reward': reward,
                'terminate': Simulator.collided}

    @staticmethod
    def on_collision():
        Simulator.collided = True

    @staticmethod
    # For RGB images.
    # Uncomment this block if you want to use RGB images. You need to set nCHANNELS=3 in constants.py
    # Comment it if you want to use Semantically Segmented images.

    # def new_frame(frame):
    #     # append the frame at the end
    #     frame = np.asarray(frame.raw_data).reshape(IMG_HEIGHT, IMG_WIDTH, 4)

    #     frame = frame[:, :, [2, 1, 0]].transpose((2, 0, 1))

    #     Simulator.frames.append(frame)

    #     while len(Simulator.frames) < nFRAMES:
    #         Simulator.frames.append(Simulator.frames[-1])

    #     # trim the list to keep only last 3 frames.
    #     Simulator.frames = Simulator.frames[-nFRAMES:]

    # For Semantically Segmented images.
    # Uncomment this block if you want to use Semantically Segmented images. You need to set nCHANNELS=1 in constants.py
    # Comment it if you want to use RGB images.
    def new_frame(frame):
        # Parse the new BGRA frame
        frame = np.asarray(frame.raw_data).reshape(IMG_HEIGHT, IMG_WIDTH, 4)

        # filter out only R channel
        frame = frame[:, :, 2]

        # Mask pixels not containing cars or roads
        mask = (frame == 7) + (frame == 10)
        frame = frame * mask
        frame = frame.astype(np.float)

        # Change pixel label to -1 for Roads
        frame[frame == 7] = -1

        # Change pixel label to 1 for Cars
        frame[frame == 10] = 1

        # Append the new frame at the last
        Simulator.frames.append(frame)

        # Duplicate the existing frames if size is less than nFRAMES
        while len(Simulator.frames) < nFRAMES:
            Simulator.frames.append(Simulator.frames[-1])

        # Trim the list to keep only last nFRAMES.
        Simulator.frames = Simulator.frames[-nFRAMES:]

    @staticmethod
    def sample_action():
        # Sample a random action
        return np.random.randint(nACTIONS)

    @staticmethod
    def state():
        # Convert the frames into array of shape (nCHANNELS * nFRAMES, IMG_HEIGHT, IMG_WIDTH)
        return np.asarray(Simulator.frames).reshape(nCHANNELS * nFRAMES, IMG_HEIGHT, IMG_WIDTH)

    def destroy(self):
        # Destroy every spawned actor
        for actor in self.actors:
            actor.destroy()

        self.actors = []
        self.vehicles = []

    def __del__(self):
        self.destroy()
