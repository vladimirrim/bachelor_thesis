# Main file. Hooks into high level world/render updates
import argparse

import experiments
from forge.trinity import smith, Trinity, Pantheon, PantheonLm, God, Sword


def parseArgs():
    parser = argparse.ArgumentParser('Projekt Godsword')
    parser.add_argument('--nRealm', type=int, default='1',
                        help='Number of environments (1 per core)')
    parser.add_argument('--ray', type=str, default='default',
                        help='Ray mode (local/default/remote)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env')
    parser.add_argument('--lm', action='store_true', default=False,
                        help='Enable lawmaker')
    return parser.parse_args()


class NativeExample:
    def __init__(self, config, args):
        trinity = Trinity(Pantheon, PantheonLm, God, Sword)
        self.env = smith.Native(config, args, trinity)

    def run(self):
        while True:
            self.env.run()


if __name__ == '__main__':
    args = parseArgs()
    config = experiments.exps['exploreIntensifiesAuto']
    # args.lm = True

    example = NativeExample(config, args)

    # Rendering by necessity snags control flow
    # This will automatically set local mode with 1 core
    if args.render:
        example.env.render()

    example.run()
