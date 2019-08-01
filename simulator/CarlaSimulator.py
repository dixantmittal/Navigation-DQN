import random
import time

import numpy

import carla
from constants import *
from simulator.ISimulator import ISimulator


# Random spawn location.
# These numbers are valid for only a particular junction scenario.
def random_location(rand):
    # Spawn on left side
    if rand < 0.5:
        return carla.Transform(carla.Location(x=random.randint(20, 60), y=random.choice([130.5, 127.5]), z=0.2), carla.Rotation(yaw=180))
    # Spawn on right side
    else:
        return carla.Transform(carla.Location(x=-random.randint(20, 60), y=random.choice([134.5, 137.5]), z=0.2), carla.Rotation(yaw=0))


class CarlaSimulator(ISimulator):
    def __init__(self, args):
        super().__init__(args)

        self.collided = True
        self.nVehicles = args.nVehicles

        self.av = None
        self.gridSensor = None
        self.collisionSensor = None
        self.vehicles = []
        self.frames = []

        # initialise Carla client
        self.world = carla.Client(args.host, args.port).get_world()
        self.bplib = self.world.get_blueprint_library()

    def reset(self):
        self.destroy()

        # Spawn Autonomous car.
        # Spawn position is random in a certain range. Numbers valid for only particular junction.
        self.av = self.world.spawn_actor(self.bplib.find('vehicle.tesla.model3'), carla.Transform(carla.Location(-6, random.randint(100, 120), 0), carla.Rotation(yaw=90)))
        # Apply brake as default control
        self.av.apply_control(carla.VehicleControl(brake=1))

        # Spawn Camera
        camera_properties = self.bplib.find('sensor.camera.semantic_segmentation')
        camera_properties.set_attribute('fov', '110')
        camera_properties.set_attribute('image_size_x', str(IMG_WIDTH))
        camera_properties.set_attribute('image_size_y', str(IMG_HEIGHT))
        camera_properties.set_attribute('sensor_tick', str(1 / FREQUENCY))
        self.gridSensor = self.world.spawn_actor(camera_properties, carla.Transform(carla.Location(0, 132, 40), carla.Rotation(roll=90, pitch=-90)), attach_to=self.av)
        self.gridSensor.listen(lambda image: self.new_frame(image))

        # Choose a random number of NPC to spawn
        vehicles_to_spawn = random.randint(1, self.nVehicles)
        tries = 0
        while vehicles_to_spawn > 0 or tries < vehicles_to_spawn * 10:

            # Spawn NPC
            vehicle = self.world.try_spawn_actor(self.bplib.find('vehicle.audi.a2'),
                                                 random_location(numpy.random.rand()))

            # if spawning failed, retry
            if vehicle is None:
                continue

            self.vehicles.append(vehicle)
            vehicles_to_spawn -= 1
            tries += 1

        # Spawn Collision detector.
        self.collisionSensor = self.world.spawn_actor(self.bplib.find('sensor.other.collision'), carla.Transform(), attach_to=self.av)
        self.collisionSensor.listen(lambda x: self.onCollision())
        self.collided = False

        # Wait for the simulator to get ready
        time.sleep(1 / FREQUENCY)

        return self.state()

    def step(self, a):
        # apply default control to NPCs
        for vehicle in self.vehicles:
            vehicle.apply_control(carla.VehicleControl(throttle=0.5))

        # apply action to AV
        self.av.apply_control(carla.VehicleControl(throttle=THROTTLE[a], brake=BRAKE[a], manual_gear_shift=True, gear=1))

        # sleep to observe action's effect
        time.sleep(1 / FREQUENCY)

        # calculate reward (\Delta s + collision penalty)
        reward = self.av.get_velocity().y * (-100 if self.collided else 1) - 1

        return self.state(), reward, self.collided, None

    def onCollision(self):
        print('Collision')
        self.collided = True

    def new_frame(self, frame):
        # Parse the new BGRA frame
        frame = numpy.asarray(frame.raw_data).reshape(IMG_HEIGHT, IMG_WIDTH, 4)

        # filter out only R channel
        frame = frame[:, :, 2]

        # Mask pixels not containing cars or roads
        mask = (frame == 7) + (frame == 10)
        frame = frame * mask
        frame = frame.astype(numpy.float)

        # Change pixel label to -1 for Roads
        frame[frame == 7] = -1

        # Change pixel label to 1 for Cars
        frame[frame == 10] = 1

        # Append the new frame at the last
        self.frames.append(frame)

        # Duplicate the existing frames if size is less than nFRAMES
        while len(self.frames) < nFRAMES:
            self.frames.append(self.frames[-1])

        # Trim the list to keep only last nFRAMES.
        self.frames = self.frames[-nFRAMES:]

    def state(self):
        # Convert the frames into array of shape (nFRAMES, IMG_HEIGHT, IMG_WIDTH)
        return numpy.reshape(self.frames, (nFRAMES, IMG_HEIGHT, IMG_WIDTH))

    def nActions(self):
        return len(THROTTLE)

    def dState(self):
        return nFRAMES, IMG_HEIGHT, IMG_WIDTH

    def sampleAction(self):
        return numpy.random.randint(self.nActions())

    def prettifyState(self, rawState):
        C, H, W = self.dState()
        return rawState.reshape(-1, C, H, W)

    def destroy(self):
        if self.collisionSensor is not None: self.collisionSensor.destroy()
        if self.gridSensor is not None: self.gridSensor.destroy()
        if self.av is not None: self.av.destroy()

        # Destroy every spawned actor
        for vehicle in self.vehicles:
            if vehicle is not None: vehicle.destroy()

        self.av = None
        self.gridSensor = None
        self.collisionSensor = None
        self.vehicles = []

    def __del__(self):
        self.destroy()
