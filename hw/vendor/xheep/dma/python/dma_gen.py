#!/usr/bin/env python3

import argparse
import hjson
import re
from pathlib import Path
from mako.template import Template

from DMA import DMA


def _to_int(value, default=0):
    if isinstance(value, str):
        return int(value, 0)
    return int(value) if value is not None else default


def create_dma_from_hjson(config_path):
    with open(config_path) as f:
        config = hjson.load(f)

    dma_cfg = config.get('dma', config)

    ch_length = _to_int(dma_cfg.get('ch_length'), 0x100)
    num_channels = _to_int(dma_cfg.get('num_channels'), 0x4)
    num_master_ports = _to_int(dma_cfg.get('num_master_ports'), 0x2)
    num_channels_per_master_port = _to_int(dma_cfg.get('num_channels_per_master_port'), 0x2)
    fifo_depth = _to_int(dma_cfg.get('fifo_depth'), 0x4)
    addr_mode = dma_cfg.get('addr_mode_en', 'yes')
    subaddr_mode = dma_cfg.get('subaddr_mode_en', 'yes')
    hw_fifo_mode = dma_cfg.get('hw_fifo_mode_en', 'yes')
    zero_padding = dma_cfg.get('zero_padding_en', 'yes')
    is_included = dma_cfg.get('is_included', 'yes')

    return DMA(
        ch_length=ch_length,
        num_channels=num_channels,
        num_master_ports=num_master_ports,
        num_channels_per_master_port=num_channels_per_master_port,
        fifo_depth=fifo_depth,
        addr_mode=addr_mode,
        subaddr_mode=subaddr_mode,
        hw_fifo_mode=hw_fifo_mode,
        zero_padding=zero_padding,
        is_included=is_included,
    )


def render_templates(dma_obj, template_dirs, output_dir):
    re_trailws = re.compile(r"[ \t\r]+$", re.MULTILINE)

    for tdir in template_dirs:
        for tpl_path in Path(tdir).glob("*.tpl"):
            template = Template(filename=str(tpl_path))
            out_path = Path(output_dir) / tpl_path.relative_to(tdir.parent)
            out_path = out_path.with_suffix('')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            code = template.render_unicode(dma=dma_obj, strict_undefined=True)
            code = re_trailws.sub("", code)
            out_path.write_text(code)


def main():
    parser = argparse.ArgumentParser(prog="dma_gen")
    parser.add_argument('--config', required=True, help='DMA HJSON config file')
    parser.add_argument('--outdir', default='.', help='Output directory')
    args = parser.parse_args()

    dma = create_dma_from_hjson(args.config)
    dma_root = Path(__file__).resolve().parent.parent
    template_dirs = [dma_root / 'data', dma_root / 'rtl']
    render_templates(dma, template_dirs, args.outdir)


if __name__ == '__main__':
    main()
