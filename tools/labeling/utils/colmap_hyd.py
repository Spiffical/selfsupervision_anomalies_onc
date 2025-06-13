import numpy as np

def colmap_hyd_py(mapsize=64, idev=1):
    iflip = 0
    if idev < 0:
        iflip = 1
        idev = abs(idev)
    
    if idev < 10:
        cmid = mapsize / 2
        np_ = int((mapsize + 1) / 5)
        top = np.ones(np_)
        bot = np.zeros(np_)
        x = np.arange(1, 2 * np_ + 2)  # Equivalent to MATLAB's 1:2*np+1

        if idev != 2:
            # Screen display
            wave = np.sin((x / max(x)) * np.pi)
            slope1 = 1.5
            slope2 = 1
        else:
            # Postscript printers
            wave = ((1 - np.cos((x / max(x)) * 2 * np.pi)) / 2) ** 1
            slope1 = 1
            slope2 = 2

        wave = wave[np_: 2 * np_]  # MATLAB's wave(np+1:2*np)
        evaw = np.flip(wave)
        red = np.concatenate([
            evaw ** slope1,
            top,
            wave ** slope2,
            bot,
            bot,
            bot
        ])
        grn = np.concatenate([
            bot,
            evaw,
            top,
            top,
            wave,
            bot
        ])
        blu = np.concatenate([
            bot,
            bot,
            bot,
            evaw ** slope2,
            top,
            wave ** slope1
        ])
        colmap0 = np.vstack((red, grn, blu)).T
        mc, nc = colmap0.shape
        dif = int((mc - mapsize) / 2)
        if dif > 0:
            cmap = np.flipud(colmap0[dif: mc - dif, :])
        else:
            cmap = np.flipud(colmap0)
        if idev == 3:
            # Fade to black at blue end
            b1 = cmap[0, 2] * 0.9
            zbk = np.linspace(b1 / 8, b1, 8)
            z = np.zeros(8)
            cmbk = np.column_stack((z, z, zbk))
            z0 = np.zeros(4)
            cmb = np.column_stack((z0, z0, z0))
            cmap = np.vstack((cmb, cmbk, cmap))
        elif idev == 4:
            # Fade to white at blue end
            mc, nc = cmap.shape
            mm = int(0.15 * mc) + 1
            cmap = cmap[int(mm / 1.5):, :]
            sgr = np.linspace(1, 0, mm)
            b1 = cmap[0, 2]
            sb = np.linspace(1, b1, mm)
            red = np.concatenate((sgr, cmap[:, 0]))
            grn = np.concatenate((sgr, cmap[:, 1]))
            blu = np.concatenate((sb, cmap[:, 2]))
            cmap = np.column_stack((red, grn, blu))
        mc, nc = cmap.shape
        cmap = np.vstack((cmap, np.array([1, 1, 1])))
    elif idev > 9:
        # Grayscale
        maxblk = 0.2
        if idev == 10:
            grey0 = np.linspace(maxblk + (1 - maxblk) / mapsize, 1, mapsize)
            cmap = np.column_stack((grey0, grey0, grey0))
        elif idev == 11:
            nc2 = int(mapsize / 2)
            grey1 = np.concatenate((
                np.linspace(maxblk + (1 - maxblk) / nc2, 1 - (1 - maxblk) / nc2, nc2),
                np.linspace(1, maxblk + (1 - maxblk) / nc2, nc2)
            ))
            cmap = np.column_stack((grey1, grey1, grey1))
            cmap = np.vstack((cmap, np.array([1.0, 1.0, 1.0])))
    else:
        cmap = np.zeros((mapsize, 3))

    if iflip == 1:
        cmap = np.flipud(cmap)

    return cmap
