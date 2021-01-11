# import retro
#
# name = 'SonicTheHedgehog-Genesis-GreenHillZone.Act1-000000.bk2'
# movie = retro.Movie(name)
# movie.step()
#
# env = retro.make(
#     game=movie.get_game(),
#     state=None,
#     # bk2s can contain any button presses, so allow everything
#     use_restricted_actions=retro.Actions.ALL,
#     players=movie.players,
# )
# env.initial_state = movie.get_state()
# env.reset()
#
# while movie.step():
#     keys = []
#     for p in range(movie.players):
#         for i in range(env.num_buttons):
#             keys.append(movie.get_key(i, p))
#     env.step(keys)
#

import imageio

src_dir = "SonicTheHedgehog-Genesis-GreenHillZone.Act1-000000.mp4"
dst_dir = "sonic.avi"

reader = imageio.get_reader(src_dir)
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer(dst_dir, fps=fps)

for im in reader:
    writer.append_data(im[:, :, :])
writer.close()