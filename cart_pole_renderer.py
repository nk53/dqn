import gym
import math
import numpy as np

class CartPoleRenderer():
    def __init__(self):
        # from cart pole's __init__()
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        #self.action_space = spaces.Discrete(2)
        #self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        #self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        # from cart pole's render()
        screen_width=600
        screen_height=400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        from gym.envs.classic_control import rendering
        self.viewer = rendering.Viewer(screen_width, screen_height)
        l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
        axleoffset =cartheight/4.0
        cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        self.carttrans = rendering.Transform()
        cart.add_attr(self.carttrans)
        self.viewer.add_geom(cart)
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        pole.set_color(.8,.6,.4)
        self.poletrans = rendering.Transform(translation=(0, axleoffset))
        pole.add_attr(self.poletrans)
        pole.add_attr(self.carttrans)
        self.viewer.add_geom(pole)
        self.axle = rendering.make_circle(polewidth/2)
        self.axle.add_attr(self.poletrans)
        self.axle.add_attr(self.carttrans)
        self.axle.set_color(.5,.5,.8)
        self.viewer.add_geom(self.axle)
        self.track = rendering.Line((0,carty), (screen_width,carty))
        self.track.set_color(0,0,0)
        self.viewer.add_geom(self.track)

        self._pole_geom = pole

    def render_observation(self, observation):
        """Sets forces this cart pole to have the state implied by
        observation, then renders it"""

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = observation
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        self.viewer.render(False)
