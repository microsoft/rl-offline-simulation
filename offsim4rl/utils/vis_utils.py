import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def plot_latent_state_color_map(df_output, output_path='latent_state.png'):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.scatter(df_output['x'], df_output['y'], c=df_output['i'], cmap='nipy_spectral', marker='.', lw=0, s=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(output_path)

def plot_episode_len_from_spinup_progress(progress_file_path, output_path='episode_length.png'):
    df = pd.read_csv(progress_file_path, sep='\t')
    plt.plot(df['EpLen'])
    plt.xlabel('epoch')
    plt.ylabel('EpLen')
    plt.savefig(output_path)

class CartPoleVisUtils():
    @staticmethod
    def plot_visited_states(visited_states, output_path='state_visitation.png'):
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 5))
        plt.plot(np.array(visited_states)[:, 0], np.array(visited_states)[:, 2], alpha=0.25, lw=0, marker='.', mew=0, markersize=3)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-0.25, 0.25)
        plt.axhline(-0.2095, c='gray')
        plt.axhline(0.2095, c='gray')
        plt.axvline(-2.4, c='gray')
        plt.axvline(2.4, c='gray')
        plt.xlabel('Cart Position')
        plt.ylabel('Pole Angle')
        plt.savefig(output_path)

    @staticmethod
    def plot_latent_state(df_output, output_path='latent_state.png'):
        plt.clf()
        plt.scatter(df_output['x'], df_output['y'], c=df_output['i'], cmap='nipy_spectral', marker='.', lw=0, s=1)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-0.25, 0.25)
        plt.xlabel('Cart Position')
        plt.ylabel('Pole Angle')
        plt.savefig(output_path)

    @staticmethod
    def render(
        state,
        terminal=False,
        render_mode='human',
        screen_width=600,
        screen_height=400,
        screen=None,
        clock=None,
        x_threshold=2.4,
        gravity=9.8,
        masscart=1.0,
        masspole=0.1,
        length=0.5,  # actually half the pole's length
        force_mag=10.0,
        tau=0.02,  # seconds between state updates
        kinematics_integrator="euler",
        render_fps=50,
    ):
        import pygame
        from pygame import gfxdraw

        total_mass = masspole + masscart,
        polemass_length = masspole * length,

        # Angle at which to fail the episode
        theta_threshold_radians = 12 * 2 * math.pi / 360

        if screen is None:
            pygame.init()
            if render_mode == "human":
                pygame.display.init()
                screen = pygame.display.set_mode(
                    (screen_width, screen_height)
                )
            else:  # mode == "rgb_array"
                screen = pygame.Surface((screen_width, screen_height))
        if clock is None:
            clock = pygame.time.Clock()

        world_width = x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * length)
        cartwidth = 50.0
        cartheight = 30.0

        x = state

        surf = pygame.Surface((screen_width, screen_height))
        if terminal:
            surf.fill((255, 0, 0))
        else:
            surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(surf, 0, screen_width, carty, (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))
        if render_mode == "human":
            pygame.event.pump()
            clock.tick(render_fps)
            pygame.display.flip()
            return screen

        elif render_mode == "rgb_array":
            return screen, np.transpose(
                np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
            )

    @staticmethod
    def replay(dataset, record_clip=False, num_frames=500, fps=20, output_dir='.'):
        os.makedirs(output_dir, exist_ok=True)
        if record_clip:
            import imageio

        frames = []
        clip_count = 0
        screen = None
        cur_episode = None
        for obs, terminal, episode_id in zip(dataset.experience['observations'], dataset.experience['terminals'], dataset.experience['episode_ids']):
            if record_clip:
                screen, frame = CartPoleVisUtils.render(obs, terminal=terminal, render_mode='rgb_array', render_fps=fps, screen=screen)
                frames.append(frame)
                if terminal:
                    for _ in range(15):
                        frames.append(frame)
            else:
                screen = CartPoleVisUtils.render(obs, terminal=terminal, render_fps=fps, screen=screen)
                if terminal:
                    time.sleep(1)

            if record_clip and len(frames) % num_frames == 0:
                imageio.mimsave(os.path.join(output_dir, f'clip_{clip_count}.gif'), frames, fps=fps)
                frames = []
                clip_count += 1

        if screen:
            import pygame

            pygame.display.quit()
            pygame.quit()
