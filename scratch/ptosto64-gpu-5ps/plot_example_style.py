"""Recreate the 04a notebook figures from this run's 10 K dipole samples."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


RUN_DIR = Path(__file__).resolve().parent
FIGURE_DIR = RUN_DIR / "figures"
PTO_LAYERS = 12
TEMPERATURE_K = 10
WINDOW = 40
DT_PS = 0.002

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "image.interpolation": "nearest",
})


def save_figure(fig, name, dpi=300):
    fig.savefig(FIGURE_DIR / name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def symmetric_norm(values):
    limit = max(float(np.percentile(np.abs(values), 99)), 1e-8)
    return TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)


def quiver_plane(ax, origin, horizontal, vertical):
    nx, ny = horizontal.shape
    x, y = np.meshgrid(
        np.arange(origin[0], origin[0] + nx),
        np.arange(origin[1], origin[1] + ny),
        indexing="ij",
    )
    magnitude = np.linalg.norm(np.stack((horizontal, vertical), axis=-1), axis=-1)
    arrow_scale = max(float(np.percentile(magnitude, 98)), 1e-8)
    ax.quiver(
        x, y, horizontal, vertical,
        angles="xy", scale_units="xy", scale=arrow_scale,
        pivot="middle", width=0.004, color="k", alpha=0.9,
    )


def most_structured_window(pz, width):
    """Choose the 40x40 window with the largest middle-PTO pz variance."""
    nx, ny = pz.shape
    if width > min(nx, ny):
        raise ValueError("Window exceeds the lateral lattice size")
    best = (-np.inf, 0, 0)
    # The original notebook visits non-overlapping windows. At L=64 that
    # gives only one candidate, so scan all positions for this reduced cell.
    for x0 in range(nx - width + 1):
        for y0 in range(ny - width + 1):
            variance = float(pz[x0:x0 + width, y0:y0 + width].var())
            if variance > best[0]:
                best = (variance, x0, y0)
    return best[1], best[2]


def load_average():
    paths = sorted(
        RUN_DIR.glob(f"sample_{TEMPERATURE_K}K_dump_*.npy"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    if not paths:
        raise FileNotFoundError("No final-temperature sample dumps found")
    shape = np.load(paths[0], mmap_mode="r").shape
    total = np.zeros(shape, dtype=np.float64)
    for path in paths:
        values = np.load(path, mmap_mode="r")
        if values.shape != shape:
            raise ValueError(f"Inconsistent dump shape: {path}")
        total += values
    return total / len(paths), paths


def plot_sections(dipole, x0, y0, norm):
    nx, ny, nz, _ = dipole.shape
    middle_layer = PTO_LAYERS // 2
    top_layer = PTO_LAYERS - 1
    window = dipole[x0:x0 + WINDOW, y0:y0 + WINDOW]
    y_cut = y0 + WINDOW // 2
    pto_origin = (nz - PTO_LAYERS) // 2
    z_center = pto_origin + (PTO_LAYERS - 1) / 2
    section = np.roll(dipole[x0:x0 + WINDOW, y_cut], pto_origin, axis=1)

    for component, name in ((2, "xz_pz.png"), (0, "xz_px.png")):
        fig, ax = plt.subplots(figsize=(4, 3.2), constrained_layout=True)
        ax.imshow(
            section[:, :, component].T,
            origin="lower",
            extent=(x0 - 0.5, x0 + WINDOW - 0.5,
                    -0.5 - z_center, nz - 0.5 - z_center),
            norm=norm, cmap="RdBu_r", aspect="equal",
        )
        quiver_plane(ax, (x0, -z_center), section[:, :, 0], section[:, :, 2])
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        save_figure(fig, name)

    for layer, name in ((middle_layer, "xy_middle.png"),
                        (top_layer, "xy_interface.png")):
        fig, ax = plt.subplots(figsize=(3, 3.1), constrained_layout=True)
        ax.imshow(
            window[:, :, layer, 2].T,
            origin="lower",
            extent=(x0 - 0.5, x0 + WINDOW - 0.5,
                    y0 - 0.5, y0 + WINDOW - 0.5),
            norm=norm, cmap="RdBu_r", aspect="equal",
        )
        quiver_plane(
            ax, (x0, y0), window[:, :, layer, 0], window[:, :, layer, 1]
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        save_figure(fig, name)

    fig, ax = plt.subplots(figsize=(0.4, 2.8))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                 cax=ax, orientation="vertical")
    ax.set_ylabel(r"$p_z$ [e$\cdot$Å]")
    save_figure(fig, "colorbar.png")
    return y_cut


def plot_system(dipole, x0, y0):
    nx, ny, nz, _ = dipole.shape
    face_step = max(1, int(np.ceil(max(nx, ny) / 700)))
    pz_box = np.asarray(dipole[::face_step, ::face_step, ::-1, 2])
    x_face = np.arange(0, nx, face_step, dtype=float)
    y_face = np.arange(0, ny, face_step, dtype=float)
    thickness = 0.05 * nx
    z_face = np.arange(nz) * thickness / (nz - 1)
    norm = symmetric_norm(pz_box[:, :, PTO_LAYERS:])
    cmap = plt.get_cmap("RdBu_r")

    def draw_face(ax, x, y, z, field):
        ax.plot_surface(
            x, y, z, facecolors=cmap(norm(field))[:-1, :-1],
            rstride=1, cstride=1, shade=False,
            linewidth=0, antialiased=False,
        )

    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111, projection="3d", proj_type="persp")
    xx, yy = np.meshgrid(x_face, y_face, indexing="ij")
    draw_face(ax, xx, yy, np.full_like(xx, z_face[-1]), pz_box[:, :, -1])
    xx, zz = np.meshgrid(x_face, z_face, indexing="ij")
    draw_face(ax, xx, np.zeros_like(xx), zz, pz_box[:, 0, :])
    yy, zz = np.meshgrid(y_face, z_face, indexing="ij")
    draw_face(ax, np.full_like(yy, x_face[-1]), yy, zz, pz_box[-1, :, :])
    outline_x = [x0, x0 + WINDOW, x0 + WINDOW, x0, x0]
    outline_y = [y0, y0, y0 + WINDOW, y0 + WINDOW, y0]
    z_mark = thickness * 1.03
    ax.plot(outline_x, outline_y, np.full(5, z_mark),
            color="k", lw=0.25, zorder=5)
    ax.view_init(elev=22, azim=-62)
    ax.set_box_aspect((nx, ny, thickness))
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, z_mark)
    ax.set_axis_off()
    save_figure(fig, "system_3d.png", dpi=800)


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    dipole, paths = load_average()
    nx, ny, nz, _ = dipole.shape
    if nz != 24 or PTO_LAYERS != 12:
        raise ValueError(f"Expected PTO12/STO12, got z={nz}")
    x0, y0 = most_structured_window(dipole[:, :, PTO_LAYERS // 2, 2], WINDOW)
    window = dipole[x0:x0 + WINDOW, y0:y0 + WINDOW]
    norm = symmetric_norm(window[:, :, :PTO_LAYERS])
    y_cut = plot_sections(dipole, x0, y0, norm)
    plot_system(dipole, x0, y0)
    first_step = int(paths[0].stem.rsplit("_", 1)[1])
    last_step = int(paths[-1].stem.rsplit("_", 1)[1])
    (FIGURE_DIR / "figure_info.txt").write_text(
        f"Cell: {nx}x{ny}x{nz}; PTO layers: z=0..{PTO_LAYERS - 1}\n"
        f"Temperature: {TEMPERATURE_K} K; {len(paths)} snapshots averaged\n"
        f"Sample time: {first_step * DT_PS:g}..{last_step * DT_PS:g} ps\n"
        f"Window: x={x0}..{x0 + WINDOW - 1}, y={y0}..{y0 + WINDOW - 1}\n"
        f"x-z cut: y={y_cut}; color range: {norm.vmin:.4g}..{norm.vmax:.4g} e Angstrom\n"
        "Cooling protocol: 300->10 K in 10 K steps, 5 ps per temperature.\n"
        "These are diagnostic figures, not equilibrated phase results.\n"
    )
    print(f"Saved six example-style figures to {FIGURE_DIR}")
    print(f"Window x={x0}..{x0+WINDOW-1}, y={y0}..{y0+WINDOW-1}; x-z cut y={y_cut}")


if __name__ == "__main__":
    main()
