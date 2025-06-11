import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def render_episode(player_replays, subsample: int = 1, replay_path: str = None):
    num_players = len(player_replays)
    if num_players == 0:
        raise ValueError("No replay found!")
    cols = min(num_players, 2)
    rows = (num_players + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), layout="tight")
    if num_players == 1:
        axs = np.array([axs])
    axs = axs.ravel()

    for i in range(num_players, len(axs)):
        axs[i].axis("off")

    fig.subplots_adjust(wspace=0, hspace=0)

    vid_frames = {}
    player_stats = {}
    for pid, replay in player_replays.items():
        # Apply subsampling to frames and game variables
        vid_frames[pid] = np.stack(replay["frames"][::subsample], 0)
        player_stats[pid] = {
            "health": [v.get("HEALTH", 0) for v in replay["game_vars"][::subsample]],
            "frags": [v.get("FRAGCOUNT", 0) for v in replay["game_vars"][::subsample]],
            "deaths": [
                v.get("DEATHCOUNT", 0) for v in replay["game_vars"][::subsample]
            ],
        }
    start_health = {pid: player_stats[pid]["health"][0] for pid in player_replays}
    start_frags = {pid: player_stats[pid]["frags"][0] for pid in player_replays}
    start_deaths = {pid: player_stats[pid]["deaths"][0] for pid in player_replays}
    imagexn = {pid: np.transpose(vid_frames[pid][0], (1, 2, 0)) for pid in vid_frames}
    plotxn = {}
    for i, (pid, img) in enumerate(imagexn.items()):
        ax = axs[i]
        plotxn[pid] = ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            pid, font="console", fontfamily="monospace", weight="bold", fontsize=20
        )
        # Add textbox for stats
        stats_text = (
            f"\u2661: {player_stats[pid]['health'][0]}\n"
            f"\u26a1: {player_stats[pid]['frags'][0]}\n"
            f"\u2620: {player_stats[pid]['deaths'][0]}"
        )
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=12,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

    def update(frame):
        for pid in plotxn:
            frame_image = np.transpose(vid_frames[pid][frame], (1, 2, 0))
            plotxn[pid].set_array(frame_image)
            # Update stats text
            stats_text = (
                f"\u2661: {player_stats[pid]['health'][frame] - start_health[pid]}\n"
                f"\u26a1: {player_stats[pid]['frags'][frame] - start_frags[pid]}\n"
                f"\u2620: {player_stats[pid]['deaths'][frame] - start_deaths[pid]}"
            )
            axs[list(player_replays.keys()).index(pid)].texts[0].set_text(stats_text)
        return list(plotxn.values())

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=min([f.shape[0] for f in vid_frames.values()]),
        interval=1000 // 35 * subsample,
        blit=True,
    )

    # Save animation to MP4 if replay_path is provided
    if replay_path:
        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        anim.save(replay_path, writer=writer)
        print(f"Animation saved to {replay_path}")

    return anim
