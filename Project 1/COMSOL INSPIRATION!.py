import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.display import HTML

# To make our code a bit cleaner we put everything in if statements, 
# this means we can "fold" the code for a specific question when we dont use it.
# As a result we have to first define some functions that we'll use in all the excercises.
# the first function makes the PML, the second and third give time and frequency plots, the fourth is for the analytical solution:

if True:
    ################################################################################################################################################
    #                                                           PML:                                                       
    ################################################################################################################################################



    def PML(D, dx , A = 10e-8, m = 4):
        #first for the velocity
        sigma_x = np.zeros_like(ox)   # shape (nx+1, ny)
        sigma_y = np.zeros_like(oy)   # shape (nx, ny+1)
        sigma_yref = np.zeros_like(oy)
        sigma_max = - (m + 1) * np.log10(A) * c / (2 * D * dx)

        for i in range(D):
            s = sigma_max * ((D-i)/D)**m
            sigma_x[i, :]      = s              
            sigma_x[-i-1, :]   = s               
            sigma_y[:, -i-1] = s 
            sigma_yref[:, -i-1] = s 
            sigma_yref[:,i] = s

        '''merk dus op dat de x-as binnen matrices verticaal is en van boven naar beneden stijgt in waarde. de y-as is horizontaal en 
        stijgt van links naar rechts in waarde. Om het uitzicht te verkrijgen zoals bij normale cartesiaanse assen, moet de martix dus
        negentig graden tegen de klok in gedraaid worden.'''

        #now for the pressure:
        # sigma_px = 0.5 * (sigma_x[:-1, :] + sigma_x[1:, :])   # shape (nx, ny)
        # sigma_py = 0.5 * (sigma_y[:, :-1] + sigma_y[:, 1:])   # shape (nx, ny)
        # sigma_pyref = 0.5 * (sigma_yref[:, :-1] + sigma_yref[:, 1:])   # shape (nx, ny)
        # sigma_p  = sigma_px + sigma_py  
        # sigma_pref = sigma_px + sigma_pyref

        '''the above method is not wrong but less accurate. As the sigma (or kappa) value varies like kappa=kappa_max*(x/d_pml)^m which is clearly not linear,
        it is better to explicitly derive the sigma_p and sigma_pref values from the distance to the PML edge (as is done below), 
        instead of using linear interpolation from the sigma values for the vector fields.'''

        sigma_px = np.zeros_like(p)   # shape (nx+1, ny)
        sigma_py = np.zeros_like(p)   # shape (nx, ny+1)
        sigma_pyref = np.zeros_like(p_ref)

        for i in range(D):
            s = sigma_max * ((D - i -1/2) / D)**4 # the -1/2 is because p is on the half integer places
            sigma_px[i, :]      = s
            sigma_px[-i-1, :]   = s
            
            sigma_py[:, -i-1]   = s
            sigma_pyref[:, -i-1] = s 
            sigma_pyref[:,i] = s

        sigma_p = sigma_px + sigma_py
        sigma_pref = sigma_px + sigma_pyref

        return sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref

    ################################################################################################################################################
    #                                                           Time/Freq plots:                                                       
    ################################################################################################################################################


    def TimePlot(t,tmax,recorders,recordersref):
        # --- time plots ---
        t = timeseries[0:tmax, 0]
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 8))

        # --- top 3: recorders with wall ---
        for i, rec in enumerate(recorders):
            axes[i].plot(t, rec[0:tmax, 0], color=f'C{i}', label=f'recorder {i+1}')
            axes[i].set_title(f'Recorder {i+1} (with wall)')
            axes[i].set_ylabel('Pressure p')
            axes[i].grid(True)

        # --- bottom: reference field (no wall) ---
        for i, rec_ref in enumerate(recordersref):
            axes[3].plot(t, rec_ref[0:tmax, 0], linestyle='--', color=f'C{i}', label=f'recorder {i+1} (ref)')

        axes[3].set_title('Reference field (no wall)')
        axes[3].set_xlabel('Time [s]')
        axes[3].set_ylabel('Pressure p')
        axes[3].legend(loc='upper right')
        axes[3].grid(True)

        plt.tight_layout()
        plt.show()

    # --- frequency analysis ---
    def freqs(recorder, recorder_ref, dt, c):
        # number of samples
        n = len(recorder)
        n_zero_pad = 10*n

        # frequency axis
        fs = 1.0 / dt
        df = fs / (n+n_zero_pad)
        freq = np.arange(n+n_zero_pad) * df
        freq[0] = 1e-5

        # wavenumber
        k = 2 * np.pi * freq / c

        # FFTs
        fft  = np.fft.fft(recorder.flatten(), n+n_zero_pad)
        fftref  = np.fft.fft(recorder_ref.flatten(), n+n_zero_pad)

        # complex ratio p / p0 and the amp/phase ratio's
        F = fft / (fftref + 1e-30)
        amp   = np.abs(F)               # amplitude ratio vs f
        phase = np.unwrap(np.angle(F))  # phase difference vs f (rad)

        return freq, k, fft, fftref, F, amp, phase


    def FreqPlot(dt,c,recorders,recordersref):
        freq, k, fft1, fft1ref, F1, amp1, phase1 = freqs(recorders[0][:,0], recordersref[0][:,0], dt, c)
        _, _, fft2, fft2ref, F2, amp2, phase2 = freqs(recorders[1][:,0], recordersref[1][:,0], dt, c)
        _, _, fft3, fft3ref, F3, amp3, phase3 = freqs(recorders[2][:,0], recordersref[2][:,0], dt, c)

        F = [amp1,amp2,amp3]
        fft_list    = [fft1, fft2, fft3]
        fftref_list = [fft1ref, fft2ref, fft3ref]

        fig, axes = plt.subplots(2, 2, sharex=True, figsize=(8, 6))
        axes = axes.ravel()   # flatten 2×2 grid to a 1D list

        # --- three recorders (with wall) ---
        for i, spec in enumerate(fft_list):
            axes[i].plot(2*np.pi/c*freq, np.abs(spec), color=f'C{i}')
            axes[i].set_title(f'Recorder {i+1} (with wall)')
            axes[i].set_ylabel('|P(k)|')
            axes[i].grid(True)
            axes[i].axvline(fc, linestyle=':', linewidth=1)
            axes[i].set_xlim(0, 2*np.pi/c*2*fc)

        # --- reference spectra on bottom-right plot ---
        for i, spec_ref in enumerate(fftref_list):
            axes[3].plot(2*np.pi/c*freq*d, np.abs(spec_ref), '--', color=f'C{i}', label=f'rec {i+1} ref')
        axes[3].set_title('Reference spectra (no wall)')
        axes[3].set_xlabel(r'$kd$ []')
        axes[3].set_ylabel('|P(kd)|')
        axes[3].grid(True)
        axes[3].legend(loc='upper right')
        axes[3].axvline(fc, linestyle=':', linewidth=1)
        
        axes[2].set_xlabel(r'$kd$ []')
        axes[3].set_xlabel(r'$kd$ []')

        plt.tight_layout()
        plt.show()

        return freq, F, fft_list, fftref_list
    
    ################################################################################################################################################
    #                                                           Analytical solution:                                                       
    ################################################################################################################################################

    from scipy.special import jv, hankel1

    def phi_and_r(rs,rr,rw,alpha=2*np.pi):
        phi_0 = np.arctan2(np.abs(rw[0]-rs[0]),np.abs(rw[1]-rs[1])) - (2*np.pi-alpha)/2
        phi = 2*np.pi - np.arctan2(np.abs(rw[0]-rr[0]),np.abs(rw[1]-rr[1])) - (2*np.pi-alpha)/2
        r0 = np.sqrt((rs[0]-rw[0])**2+(rs[1]-rw[1])**2)
        r = np.sqrt((rr[0]-rw[0])**2+(rr[1]-rw[1])**2)
        return phi_0,phi,r0,r

    def analytical_solution(k,rs,rr,rw,alpha=2*np.pi,n=50): 
        
        nu = np.pi/alpha
        eps = [1/2,1] 
        
        phi_0,phi,r0,r = phi_and_r(rs,rr,rw,alpha) # no mirroring
        
        uh = 0
        if r<r0:
            for l in range(n):
                uh+=eps[1*(l>0)]*jv(nu*l,k*r)*hankel1(nu*l,k*r0)*np.cos(nu*l*phi_0)*np.cos(nu*l*phi)
        else:
            for l in range(n):
                uh+=eps[1*(l>0)]*jv(nu*l,k*r0)*hankel1(nu*l,k*r)*np.cos(nu*l*phi_0)*np.cos(nu*l*phi)
                
        phi_0,phi,r0,r = phi_and_r([rs[0],-rs[1]],rr,rw,alpha) # mirror source
        
        if r<r0:
            for l in range(n):
                uh+=eps[1*(l>0)]*jv(nu*l,k*r)*hankel1(nu*l,k*r0)*np.cos(nu*l*phi_0)*np.cos(nu*l*phi)
        else:
            for l in range(n):
                uh+=eps[1*(l>0)]*jv(nu*l,k*r0)*hankel1(nu*l,k*r)*np.cos(nu*l*phi_0)*np.cos(nu*l*phi)
                
        phi_0,phi,r0,r = phi_and_r(rs,[rr[0],-rr[1]],rw,alpha) # mirror receiver
        
        if r<r0:
            for l in range(n):
                uh+=eps[1*(l>0)]*jv(nu*l,k*r)*hankel1(nu*l,k*r0)*np.cos(nu*l*phi_0)*np.cos(nu*l*phi)
        else:
            for l in range(n):
                uh+=eps[1*(l>0)]*jv(nu*l,k*r0)*hankel1(nu*l,k*r)*np.cos(nu*l*phi_0)*np.cos(nu*l*phi)
        
        phi_0,phi,r0,r = phi_and_r([rs[0],-rs[1]],[rr[0],-rr[1]],rw,alpha) # mirror source and receiver
        
        if r<r0:
            for l in range(n):
                uh+=eps[1*(l>0)]*jv(nu*l,k*r)*hankel1(nu*l,k*r0)*np.cos(nu*l*phi_0)*np.cos(nu*l*phi)
        else:
            for l in range(n):
                uh+=eps[1*(l>0)]*jv(nu*l,k*r0)*hankel1(nu*l,k*r)*np.cos(nu*l*phi_0)*np.cos(nu*l*phi)
        
        u0 = 1j/4 * hankel1(0,k*np.sqrt((rr[0]-rs[0])**2+(rr[1]-rs[1])**2)) # free field
        
        return uh*np.pi/(alpha*1j) / u0


################################################################################################################################################
#                                                           Question 1:                                                       
################################################################################################################################################


antwoord = input("Do you want to see our thin wall? type 'y' to continue: ")
boolse=(antwoord.lower() == 'y')
if boolse:
    ################################################################################################################################################
    #                                                           Initialization:                                                       
    ################################################################################################################################################



    c = 340 # m/s
    d = 1 # m
    kd = 5 # k = 2pi/lambda 
    if True: #Set 'True' if you wan tot pick kd
        kd = float(input(r'Give value for k_sd:'))

    dx = d/20
    dy = dx

    nx = int(6*d/dx) # total number of x cells
    N = int(3/2*d/dy) # total whitespace cells for free field beneath surface 
    ny = int(3*d/dy + N) # # total number of y cells

    CFL = 1 # courant number, high CFL means accurate but less stable (max CFL = 1)

    dt = CFL/(c*np.sqrt((1/dx**2)+(1/dy**2))) # from the courant number inequality
    nt = int(10*nx/CFL) # total number of timesteps

    x0 = int(d/dx) # x index of source
    y0 = int((d/10)/dy + N) # y index of source
    Zs = 2*c

    A=1
    fc= kd/(2*np.pi) * c / (d)
    a = 3
    if True: #Set 'True' if you wan tot pick a
        a = float(input('Give value for a:'))
    sigma=a*d/(c*kd)
    t0 = 3*sigma

    # The source used in this project is defined as:
    # source = A*np.cos(2*np.pi*fc*(t-t0))*np.exp(-1/2*(t-t0)**2)/(sigma**2))

    #observation points:
    x1 = int(x0 + 2*d/dx)
    y1 = int((d/2)/dy + N)

    x2 = int(x1 + d/dx)
    y2 = y1

    x3 = int(x2 + d/dx)
    y3 = y2

    #initialize fields
    ox = np.zeros((nx+1, ny))
    ox_ref = ox.copy()
    oy = np.zeros((nx, ny+1))
    oy_ref = oy.copy()
    p = np.zeros((nx, ny)) 
    p_ref = p.copy()

    #wall & ground:
    i_wall = int(x0 + d/dx) # x coord
    j_wall = int(N + 2*d/dy) # y coord
    j_floor = int(N)

    #total simulation time: when the pulse from the mirror source has travelled to the furthest mirror detector
    R = np.sqrt((d)**2 + (2*d+d/10)**2) + np.sqrt((3*d)**2 + (2*d+d/2)**2)
    nt = int((t0 + 5*sigma + R/c) * CFL *1.2/ dt)
    
    timeseries = np.zeros((nt,1))
    
    recorder1 = np.zeros((nt,1))
    recorder1_ref = np.zeros((nt,1))

    recorder2 = np.zeros((nt,1))
    recorder2_ref = np.zeros((nt,1))

    recorder3 = np.zeros((nt,1))
    recorder3_ref = np.zeros((nt,1))
    
    tmax = nt
    
    ################################################################################################################################################
    #                                                           Updater                                                       
    ################################################################################################################################################



    sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref = PML(dx = dx, D= int((3*d/4)/dx)) 

    def updater(c,dx,dy,dt):
        global p,ox,oy,p_ref,ox_ref,oy_ref,sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref

        # ox has shape (nx+1, ny), oy has (nx, ny+1)
        dpx = np.append(p,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p,axis =0)
        dpy = np.append(p,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p,axis =1)
        dpx_ref = np.append(p_ref,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p_ref,axis =0)
        dpy_ref = np.append(p_ref,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p_ref,axis =1)

        #call in PML:
        ox = ((1 - (sigma_x*dt)/2) * ox - dt * dpx / dx)/(1 + (sigma_x*dt)/2)
        oy = ((1 - (sigma_y*dt)/2) * oy - dt * dpy / dy)/(1 + (sigma_y*dt)/2)
        ox_ref = ((1 - (sigma_x*dt)/2) * ox_ref - dt * dpx_ref / dx)/(1 + (sigma_x*dt)/2)
        oy_ref = ((1 - (sigma_yref*dt)/2) * oy_ref - dt * dpy_ref / dy)/(1 + (sigma_yref*dt)/2)

        #Wall,  normal is in x direction so ox = 0
        ox[i_wall, : j_wall+1] = 0.0
        #Floor, normal is in y direction so oy = 0
        oy[:, N] = 0

        dOx = ox[1:, :] - ox[:-1, :]
        dOy = oy[:, 1:] - oy[:, :-1]
        dOx_ref = ox_ref[1:, :] - ox_ref[:-1, :]
        dOy_ref = oy_ref[:, 1:] - oy_ref[:, :-1]

        p = ((1 - (sigma_p * dt)/2) * p - c**2 * dt * (dOx/dx + dOy/dy))/(1 + (sigma_p * dt)/2)
        p_ref = ((1 - (sigma_pref * dt)/2) * p_ref - c**2 * dt * (dOx_ref/dx + dOy_ref/dy))/(1 + (sigma_pref * dt)/2)


    ################################################################################################################################################
    #                                                           Plots:                                                       
    ################################################################################################################################################



    #---- plot of the animation ----
    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.xlim([1, nx+1])
    plt.ylim([1, ny+1])
    movie = []

    for it in range(0, nt):
        t = (it-1)*dt
        timeseries[it, 0] = t
        print('%d/%d' % (it, nt))

        bron = A*np.cos(2*np.pi*fc*(t-t0))*np.exp(-1/2*((t-t0)/sigma)**2) # update source

        p[x0,y0] = p[x0,y0] + bron
        p_ref[x0,y0] = p_ref[x0,y0] + bron # add source
        updater(c,dx,dy,dt)   # propagate over dt

        recorder1[it] = p[x1,y1] # store p field at recorder
        recorder1_ref[it] = p_ref[x1,y1] # store reference p field at recorder

        recorder2[it] = p[x2,y2]
        recorder2_ref[it] = p_ref[x2,y2]

        recorder3[it] = p[x3,y3]
        recorder3_ref[it] = p_ref[x3,y3]

        artists = [
            ax.text(0.5,1.05,'%d/%d' % (it, nt), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, ),
            ax.imshow(p.T, vmin=-0.02*A, vmax=0.02*A),
            # ax.imshow(p_ref.T, vmin=-0.02*A, vmax=0.02*A),
            ax.plot([i_wall, i_wall], [N, j_wall], 'k-', linewidth=1)[0],
            ax.plot([0, nx], [N,N], 'k-', linewidth=1)[0],
            ax.plot(x0,y0,'ks',fillstyle="none")[0],
            ax.plot(x1,y1,'ro',fillstyle="none")[0],
            ax.plot(x2,y2,'ro',fillstyle="none")[0],
            ax.plot(x3,y3,'ro',fillstyle="none")[0],
            ]
        movie.append(artists)
    my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000,
                                    blit=True)

    recorders    = [recorder1, recorder2, recorder3]
    recordersref = [recorder1_ref, recorder2_ref, recorder3_ref]
    plt.show() 



    TimePlot(t,tmax,recorders,recordersref)
    freq, F, fftlist, fftreflist = FreqPlot(dt,c,recorders,recordersref)



    

    k = np.linspace(0.1,np.pi/dx*1.1,nt)/d
    u = [analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x1*dx,(y1-N)*dy]),np.array([i_wall*dx,(j_wall-N)*dy])),
        analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x2*dx,(y2-N)*dy]),np.array([i_wall*dx,(j_wall-N)*dy])),
        analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x3*dx,(y3-N)*dy]),np.array([i_wall*dx,(j_wall-N)*dy]))]

    

    #now plot numerical AND analytical
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    axes = axes.ravel()
    
    
    for i,e in enumerate(F):
        axes[i].plot(freq*2*np.pi/c*d,e, color=f'C{i}',label='Numerical calculation')
        axes[i].plot(k*d,np.abs(u[i]),"--",color=f'C{i}',label='Analytical solution')
        # axes[i].axvline(np.pi/dx,0,1.5,color="red",label=r'Sampling $k_{max}$')
        axes[i].axvline((2*np.pi/c*fc*d+3/(c*sigma))*d,0,1.5,color="red")
        axes[i].axvline((2*np.pi/c*fc*d-3/(c*sigma))*d,0,1.5,color="red")
        # axes[i].axvline((2*np.pi/c*fc*d+3/(c*sigma))*d,0,1.5,color="red",label=r'Source $k_{max}$')
        # axes[i].axvline((2*np.pi/c*fc*d-3/(c*sigma))*d,0,1.5,color="blue",label=r'Source $k_{min}$')
        axes[i].grid(True)
        axes[i].set_xlim(0,10)
        axes[i].set_ylim(0,1.5)
        axes[i].set_title(f'Recorder {i+1}')
        axes[i].legend()

    axes[-1].set_xlabel('kd []')
    plt.tight_layout()
    plt.show()





################################################################################################################################################
#                                                           Question 2:                                                       
################################################################################################################################################



answer = input("Do you want to see our thick wall? type 'y' to continue: ")
boolean=(answer.lower() == 'y')
if boolean:
    ################################################################################################################################################
    #                                                           Initialization:                                                       
    ################################################################################################################################################

    #New geometry but we keep a lot of the old stuff:
    c = 340 # m/s
    d = 2 # m
    kd = 10 # k = 2pi/lambda 
    if True: #Set 'True' if you wan tot pick kd
        kd = float(input(r'Give value for k_sd:'))
    
    Zs= 2*c
    if True: #Set 'True' if you wan tot pick Zs as a multiple of c
        Zs = float(input('Give value for Zs/c:'))*c
    
    CFL = 1
    dx = d/40
    dy = dx
    
    nx = int(7*d/dx) # total number of x cells
    N = int(3/2*d/dy) # total whitespace cells for free field beneath surface 
    ny = int(3*d/dy + N) # # total number of y cells

    dt = CFL/(c*np.sqrt((1/dx**2)+(1/dy**2))) # from the courant number inequality

    A=1
    fc= kd/(2*np.pi) * c / (d)
    a = 3
    sigma=a*d/(c*kd)
    t0 = 3*sigma

    x0 = int(d/dx)
    y0 = int(N + (d/10)/dy) 

    # Wall: vertical block of width d, height 2d
    i_obj_left  = int(2*d/dx) 
    i_obj_right = int(3*d/dx) 
    j_obj_top   = int(N + 2*d/dy)     

    #recorders:
    x1 = int(i_obj_right + d/dx)
    x2 = int(x1 + d/dx)
    x3 = int(x2 + d/dx)
    y1 = int(N + (d/2)/dy)
    y2 = y1
    y3 = y1

    #total simulation time: when the pulse from the mirror source has travelled to the furthest mirror detector
    R = np.sqrt((d)**2 + (2*d+d/10)**2) + d + np.sqrt((3*d)**2 + (2*d+d/2)**2)
    nt = int((t0 + 5*sigma + R/c) * CFL *1.2 / dt)

    timeseries    = np.zeros((nt, 1))
    
    recorder1     = np.zeros((nt, 1))
    recorder1_ref = np.zeros((nt, 1))
    recorder1_0 = np.zeros((nt, 1))
    
    recorder2     = np.zeros((nt, 1))
    recorder2_ref = np.zeros((nt, 1))
    recorder2_0 = np.zeros((nt, 1))
    
    recorder3     = np.zeros((nt, 1))
    recorder3_ref = np.zeros((nt, 1))
    recorder3_0 = np.zeros((nt, 1))
    
    tmax = nt

    #initialize:

    ox     = np.zeros((nx+1, ny))   # Zc = 2
    ox_ref = ox.copy()              # without ground or wall
    ox_0   = ox.copy()              # Zc = infinity
    oy     = np.zeros((nx, ny+1))
    oy_ref = oy.copy()
    oy_0   = oy.copy()
    p      = np.zeros((nx, ny))
    p_ref  = p.copy()
    p_0    = p.copy()

    sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref = PML(dx = dx, D = int(3*d/(4*dx))) 

    def updater2(c,dx,dy,dt):
        global p,ox,oy,p_ref,ox_ref,oy_ref,p_0,ox_0,oy_0,sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref

        # ox has shape (nx+1, ny), oy has (nx, ny+1)
        dpx = np.append(p,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p,axis =0)
        dpy = np.append(p,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p,axis =1)
        dpx_ref = np.append(p_ref,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p_ref,axis =0)
        dpy_ref = np.append(p_ref,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p_ref,axis =1)
        dpx_0 = np.append(p_0,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p_0,axis =0)
        dpy_0 = np.append(p_0,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p_0,axis =1)

        ox_old = ox.copy() 
        oy_old = oy.copy()

        # PML:
        ox = ((1 - (sigma_x*dt)/2) * ox - dt * dpx / dx)/(1 + (sigma_x*dt)/2)
        oy = ((1 - (sigma_y*dt)/2) * oy - dt * dpy / dy)/(1 + (sigma_y*dt)/2)
        ox_ref = ((1 - (sigma_x*dt)/2) * ox_ref - dt * dpx_ref / dx)/(1 + (sigma_x*dt)/2)
        oy_ref = ((1 - (sigma_yref*dt)/2) * oy_ref - dt * dpy_ref / dy)/(1 + (sigma_yref*dt)/2)
        ox_0 = ((1 - (sigma_x*dt)/2) * ox_0 - dt * dpx_0 / dx)/(1 + (sigma_x*dt)/2)
        oy_0 = ((1 - (sigma_y*dt)/2) * oy_0 - dt * dpy_0 / dy)/(1 + (sigma_y*dt)/2)
        
        Ax = Zs * dt / dx
        Ay = Zs * dt / dy

        # Left wall
        ox[i_obj_left, N:j_obj_top+1] = (
            (1 - Ax) * ox_old[i_obj_left, N:j_obj_top+1]
            + 2 * dt / dx * p[i_obj_left-1, N:j_obj_top+1]) / (1 + Ax)
        
        ox_0[i_obj_left, N:j_obj_top+1] = 0

        # Right wall
        ox[i_obj_right, N:j_obj_top+1] = (
            (1 - Ax) * ox_old[i_obj_right, N:j_obj_top+1]
            - 2 * dt / dx * p[i_obj_right, N:j_obj_top+1]) / (1 + Ax)
        ox_0[i_obj_right, N:j_obj_top+1] = 0

        # Top wall
        oy[i_obj_left:i_obj_right+1, j_obj_top] = (
            (1 - Ay) * oy_old[i_obj_left:i_obj_right+1, j_obj_top]
            - 2 * dt / dy * p[i_obj_left:i_obj_right+1, j_obj_top]) / (1 + Ay)
        oy_0[i_obj_left:i_obj_right+1, j_obj_top] = 0
        
        

        # now we need to set p = 0 in the wall to ensure no propagation inside the wall:
        # this is done at the very end of this updater function
        
        # floor: normal in y direction -> oy = 0
        oy[:, N] = 0
        oy_0[:, N] = 0

        dOx = ox[1:, :] - ox[:-1, :]
        dOy = oy[:, 1:] - oy[:, :-1]
        
        dOx_ref = ox_ref[1:, :] - ox_ref[:-1, :]
        dOy_ref = oy_ref[:, 1:] - oy_ref[:, :-1]
        
        dOx_0 = ox_0[1:, :] - ox_0[:-1, :]
        dOy_0 = oy_0[:, 1:] - oy_0[:, :-1]

        p = ((1 - (sigma_p * dt)/2) * p - c**2 * dt * (dOx/dx + dOy/dy))/(1 + (sigma_p * dt)/2)
        p_ref = ((1 - (sigma_pref * dt)/2) * p_ref - c**2 * dt * (dOx_ref/dx + dOy_ref/dy))/(1 + (sigma_pref * dt)/2)
        p_0 = ((1 - (sigma_p * dt)/2) * p_0 - c**2 * dt * (dOx_0/dx + dOy_0/dy))/(1 + (sigma_p * dt)/2)
                
        p[i_obj_left, N:j_obj_top] = 0.0
        p[i_obj_right - 1, N:j_obj_top] = 0.0
        p[i_obj_left:i_obj_right, j_obj_top-1] = 0.0  
        
        

    #---- plot of the animation ----
    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.xlim([1, nx+1])
    plt.ylim([1, ny+1])

    showplot = input("Type 'absorbing', 'free', or 'perfect' to show their respective animation:")
    showplot_i = 0*(showplot == "absorbing") + 1*(showplot == "free") + 2*(showplot == "perfect")
    
    movie = []

    for it in range(0, nt):
        t = (it-1)*dt
        timeseries[it, 0] = t
        print('%d/%d' % (it, nt))

        source = A*np.cos(2*np.pi*fc*(t-t0))*np.exp(-1/2*((t-t0)/sigma)**2) # update source

        p[x0,y0] = p[x0,y0] + source
        p_ref[x0,y0] = p_ref[x0,y0] + source # add source
        p_0[x0,y0] = p_0[x0,y0] + source
        updater2(c,dx,dy,dt)   # propagate over dt

        recorder1[it] = p[x1,y1] # store p field at recorder
        recorder1_ref[it] = p_ref[x1,y1] # store reference p field at recorder
        recorder1_0[it] = p_0[x1,y1]

        recorder2[it] = p[x2,y2]
        recorder2_ref[it] = p_ref[x2,y2]
        recorder2_0[it] = p_0[x2,y2]
        
        recorder3[it] = p[x3,y3]
        recorder3_ref[it] = p_ref[x3,y3]
        recorder3_0[it] = p_0[x3,y3]
        
        artists = [
            ax.text(0.5,1.05,'%d/%d' % (it, nt), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, ),
            ax.imshow([p.T,p_ref.T,p_0.T][showplot_i], vmin=-0.02*A, vmax=0.02*A),
            ax.plot([i_obj_left, i_obj_left], [N, j_obj_top], 'k-', linewidth=1)[0],
            ax.plot([i_obj_left, i_obj_right], [j_obj_top, j_obj_top], 'k-', linewidth=1)[0],
            ax.plot([i_obj_right, i_obj_right], [N, j_obj_top], 'k-', linewidth=1)[0],
            ax.plot([0, nx], [N,N], 'k-', linewidth=1)[0],
            ax.plot(x0,y0,'ks',fillstyle="none")[0],
            ax.plot(x1,y1,'ro',fillstyle="none")[0],
            ax.plot(x2,y2,'ro',fillstyle="none")[0],
            ax.plot(x3,y3,'ro',fillstyle="none")[0],
            ]
        movie.append(artists)
    my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000,
                                    blit=True)

    plt.show()

    recorders    = [recorder1, recorder2, recorder3]
    recordersref = [recorder1_ref, recorder2_ref, recorder3_ref]
    recorders0 = [recorder1_0, recorder2_0, recorder3_0]


    TimePlot(t,tmax,recorders,recordersref)
    freq, F, fftlist, fftreflist = FreqPlot(dt,c,recorders,recordersref)
    TimePlot(t,tmax,recorders0,recordersref)
    freq0, F0, fftlist0, fftreflist0 = FreqPlot(dt,c,recorders0,recordersref)
    
    #This is the analytical solution for the thin wall, only used to explore the effect of a thick wall
    k = np.linspace(0.1,np.pi/dx*1.1,nt)/d
    u = [analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x0*dx+2*d,(y1-N)*dy]),np.array([x0*dx+d,(j_obj_top-N)*dy])),
        analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x0*dx+3*d,(y2-N)*dy]),np.array([x0*dx+d,(j_obj_top-N)*dy])),
        analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x0*dx+4*d,(y3-N)*dy]),np.array([x0*dx+d,(j_obj_top-N)*dy]))]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    axes = axes.ravel()
    for i,e in enumerate(F):
        # axes[i].plot(freq*2*np.pi/c * d,F[i],"--", color=f'C{i}',label=f"Zs = {int(Zs/c)}c")
        axes[i].plot(freq*2*np.pi/c * d,F0[i], color=f'C{i}',ls='-',label=[r"Zs = $\infty$",'Numerical solution thick wall'][1])
        axes[i].plot(k*d,np.abs(u[i]),"--",color=f'C{i}',label='Analytical solution thin wall')
        axes[i].grid(True)
        axes[i].set_xlim(0,10)
        axes[i].set_ylim(0,1.5)
        axes[i].set_title(f'Recorder {i+1}')
        axes[i].legend()
    plt.show()



################################################################################################################################################
#                                                           Question 3, staircasing:                                                       
################################################################################################################################################



answer = input("Do you want to see our triangular wall (staircasing implementation)? type 'y' to continue: ")
boolean=(answer.lower() == 'y')
if boolean:
    ################################################################################################################################################
    #                                                           Initialization:                                                       
    ################################################################################################################################################

    #New geometry but we keep a lot of the old stuff:
    c = 340 # m/s
    d = 2 # m
    kd = 5 # k = 2pi/lambda 
    if True: #Set 'True' if you wan tot pick kd
        kd = float(input(r'Give value for k_sd:'))
    
    Zs= 2*c
    if False: #Set 'True' if you wan tot pick Zs
        Zs = float(input('Give value for Zs/c:'))*c
    CFL = 1

    dx = d/20
    dy = dx
    nx = int(7*d/dx) # total number of x cells
    N = int(3/2*d/dy) # total whitespace cells for free field beneath surface 
    ny = int(3*d/dy + N) # # total number of y cells

    dt = CFL/(c*np.sqrt((1/dx**2)+(1/dy**2))) # from the courant number inequality

    A=1
    fc= kd/(2*np.pi) * c / (d)
    a = 3
    sigma=a*d/(c*kd)
    t0 = 3*sigma

    x0 = int(d/dx)
    y0 = int(N + (d/10)/dy) 


    # We make a mask mask_triangle with the same shape as p, where we select which cells belong to the triangle:
    # triangle with base d and height 2d:
    aoe = round(2*int(d /dx) / 4) - 1 #the amount of extentions, or stairs, that we will have
    base = 2*aoe + 1 #number of cells at the base. This amount has been chosen in such a way, that
    # the triangle will end up having a 'pointy' tip, that means, only one p value wide, so that 
    # it is certain diffraction will occur only once. 

    mask_p = np.zeros((nx, ny))

    ix_start = int(x0 + int(d /dx)) 
    ix_end = ix_start + base - 1

    for j in range(N, N + 4*(aoe + 1)):
        mask_p[ix_start + ((j - N) // 4):ix_end - ((j - N) // 4) + 1, j] = 1

    # generate mask of ox:
    mask_ox = np.zeros((nx+1, ny))

    ix_start = int(x0 + int(d /dx)) 
    ix_end = ix_start + base - 1 + 1

    for j in range(N, N + 4*(aoe + 1)):
        mask_ox[ix_start + ((j - N) // 4), j] = 1
        mask_ox[ix_end - ((j - N) // 4), j] = 1

    # For the impedance calculation it is important to split the ox mask in two
    # one hase the components on the left side of the triangle, the other on the right side

    mask_ox_left = np.zeros((nx+1, ny))

    ix_start = int(x0 + int(d /dx)) 

    for j in range(N, N + 4*(aoe + 1)):
        mask_ox_left[ix_start + ((j - N) // 4), j] = 1

    mask_ox_right = np.zeros((nx+1, ny))

    ix_end = ix_start + base - 1 + 1

    for j in range(N, N + 4*(aoe + 1)):
        mask_ox_right[ix_end - ((j - N) // 4), j] = 1



    mask_oy= np.zeros((nx, ny+1))

    ix_start = int(x0 + int(d /dx))  
    ix_end = ix_start + base - 1 

    for j in range(1,aoe+2):
        mask_oy[ix_start , N + 4*j] = 1
        mask_oy[ix_end , N + 4*j] = 1
        ix_start+=1
        ix_end-=1

    i_obj_right = int(x0+2*int(d/dx))
    #recorders:
    x1 = int(i_obj_right + int(d/dx))
    x2 = int(x1 + int(d/dx))
    x3 = int(x2 + int(d/dx))
    y1 = int(N + int(d/dy)/2)
    y2 = y1
    y3 = y1
    
    #total simulation time: when the pulse from mirror source has travelled to the furthest mirror detector
    R = np.sqrt((dx*(i_obj_right-x0))**2 + (2*d+d/10)**2) + np.sqrt((dx*(i_obj_right-x3))**2 + (2*d+d/2)**2)
    nt = int((t0 + 5*sigma + R/c) * CFL *1.2/ dt)
    
    tmax = nt

    #initialize:
    timeseries     = np.zeros((nt, 1))
    recorder1     = np.zeros((nt, 1))
    recorder1_ref = np.zeros((nt, 1))
    recorder1_0     = np.zeros((nt, 1))
    recorder2     = np.zeros((nt, 1))
    recorder2_ref = np.zeros((nt, 1))
    recorder2_0     = np.zeros((nt, 1))
    recorder3     = np.zeros((nt, 1))
    recorder3_ref = np.zeros((nt, 1))
    recorder3_0     = np.zeros((nt, 1))

    ox     = np.zeros((nx+1, ny))
    ox_ref = ox.copy()
    ox_0   = ox.copy()
    oy     = np.zeros((nx, ny+1))
    oy_ref = oy.copy()
    oy_0   = oy.copy()
    p      = np.zeros((nx, ny))
    p_ref  = p.copy()
    p_0    = p.copy()

    sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref = PML(dx = dx, D = int(3*d/(4*dx))) 

    

    def updater3(c,dx,dy,dt):
        global p,ox,oy,p_ref,ox_ref,oy_ref,p_0,ox_0,oy_0,sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref


        # ox has shape (nx+1, ny), oy has (nx, ny+1)
        dpx = np.append(p,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p,axis =0)
        dpy = np.append(p,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p,axis =1)
        dpx_ref = np.append(p_ref,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p_ref,axis =0)
        dpy_ref = np.append(p_ref,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p_ref,axis =1)
        dpx_0 = np.append(p_0,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p_0,axis =0)
        dpy_0 = np.append(p_0,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p_0,axis =1)

        ox_old = ox.copy() 
        oy_old = oy.copy()


        # PML:
        ox = ((1 - (sigma_x*dt)/2) * ox - dt * dpx / dx)/(1 + (sigma_x*dt)/2)
        oy = ((1 - (sigma_y*dt)/2) * oy - dt * dpy / dy)/(1 + (sigma_y*dt)/2)
        ox_ref = ((1 - (sigma_x*dt)/2) * ox_ref - dt * dpx_ref / dx)/(1 + (sigma_x*dt)/2)
        oy_ref = ((1 - (sigma_yref*dt)/2) * oy_ref - dt * dpy_ref / dy)/(1 + (sigma_yref*dt)/2)
        ox_0 = ((1 - (sigma_x*dt)/2) * ox_0 - dt * dpx_0 / dx)/(1 + (sigma_x*dt)/2)
        oy_0 = ((1 - (sigma_y*dt)/2) * oy_0 - dt * dpy_0 / dy)/(1 + (sigma_y*dt)/2)
        

        # surface impedance:
        Ax = Zs * dt / dx
        Ay = Zs * dt / dy

        # ____Left ox components____
        idx = (mask_ox_left == 1) # we convert to boolean values

        i_idx, j_idx = np.where(idx)

        # Update ox only on the masked positions
        ox[i_idx, j_idx] = (
            (1 - Ax) * ox_old[i_idx, j_idx]
            + (2 * dt / dx) * p[i_idx - 1, j_idx]
        ) / (1 + Ax)
        ox_0[i_idx, j_idx] = 0

        # ___Right ox components___
        idx = (mask_ox_right == 1) # we convert to boolean values

        i_idx, j_idx = np.where(idx)

        # Update ox only on the masked positions
        ox[i_idx, j_idx] = (
            (1 - Ax) * ox_old[i_idx, j_idx]
            - (2 * dt / dx) * p[i_idx, j_idx]
        ) / (1 + Ax)
        ox_0[i_idx, j_idx] = 0

        # ___oy components___
        idy = (mask_oy == 1) # we convert to boolean values

        i_idy, j_idy = np.where(idy)

        # Update oy only on the masked positions
        oy[i_idy, j_idy] = (
            (1 - Ay) * oy_old[i_idy, j_idy]
            - (2 * dt / dx) * p[i_idy, j_idy]
        ) / (1 + Ay)
        oy_0[i_idy, j_idy] = 0

        # now we need to set p = 0 in the wall to ensure no propagation inside the wall.
        # this is done at the very end of the updater.


        # floor: normal in y direction -> oy = 0
        oy[:, N] = 0
        oy_0[:, N] = 0

        dOx = ox[1:, :] - ox[:-1, :]
        dOy = oy[:, 1:] - oy[:, :-1]
        dOx_ref = ox_ref[1:, :] - ox_ref[:-1, :]
        dOy_ref = oy_ref[:, 1:] - oy_ref[:, :-1]
        dOx_0 = ox_0[1:, :] - ox_0[:-1, :]
        dOy_0 = oy_0[:, 1:] - oy_0[:, :-1]

        p = ((1 - (sigma_p * dt)/2) * p - c**2 * dt * (dOx/dx + dOy/dy))/(1 + (sigma_p * dt)/2)   
        p_ref = ((1 - (sigma_pref * dt)/2) * p_ref - c**2 * dt * (dOx_ref/dx + dOy_ref/dy))/(1 + (sigma_pref * dt)/2)
        p_0 = ((1 - (sigma_p * dt)/2) * p_0 - c**2 * dt * (dOx_0/dx + dOy_0/dy))/(1 + (sigma_p * dt)/2)
        
        p *= (1-mask_p)
        p_0 *= (1-mask_p)

    #---- plot of the animation ----
    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.xlim([1, nx+1])
    plt.ylim([1, ny+1])

    showplot = input("Type 'absorbing', 'free', or 'perfect' to show their respective animation:")
    showplot_i = 0*(showplot == "absorbing") + 1*(showplot == "free") + 2*(showplot == "perfect")

    movie = []

    for it in range(0, nt):
        t = (it-1)*dt
        timeseries[it, 0] = t
        print('%d/%d' % (it, nt))

        source = A*np.cos(2*np.pi*fc*(t-t0))*np.exp(-1/2*((t-t0)/sigma)**2) # update source

        p[x0,y0] = p[x0,y0] + source
        p_ref[x0,y0] = p_ref[x0,y0] + source # add source
        p_0[x0,y0] = p_0[x0,y0] + source
        updater3(c,dx,dy,dt)   # propagate over dt

        recorder1[it] = p[x1,y1] # store p field at recorder
        recorder1_ref[it] = p_ref[x1,y1] # store reference p field at recorder
        recorder1_0[it] = p_0[x1,y1]

        recorder2[it] = p[x2,y2]
        recorder2_ref[it] = p_ref[x2,y2]
        recorder2_0[it] = p_0[x2,y2]

        recorder3[it] = p[x3,y3]
        recorder3_ref[it] = p_ref[x3,y3]
        recorder3_0[it] = p_0[x3,y3]

        artists = [
            ax.text(0.5,1.05,'%d/%d' % (it, nt), 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes, ),
            ax.imshow([p.T,p_ref.T,p_0.T][showplot_i], vmin=-0.02*A, vmax=0.02*A),
            ax.contour(mask_p.T.astype(float), levels=[0.5], colors='k', linewidths=1),
            ax.plot([0, nx], [N, N], 'k-', linewidth=1)[0],
            ax.plot(x0, y0, 'ks', fillstyle="none")[0],
            ax.plot(x1, y1, 'ro', fillstyle="none")[0],
            ax.plot(x2, y2, 'ro', fillstyle="none")[0],
            ax.plot(x3, y3, 'ro', fillstyle="none")[0],
            ax.plot([0, nx], [N,N], 'k-', linewidth=1)[0],
            ax.plot(x0,y0,'ks',fillstyle="none")[0],
            ax.plot(x1,y1,'ro',fillstyle="none")[0],
            ax.plot(x2,y2,'ro',fillstyle="none")[0],
            ax.plot(x3,y3,'ro',fillstyle="none")[0],
            ]
        movie.append(artists)
    my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000, blit=True)

    plt.show()

    recorders    = [recorder1, recorder2, recorder3]
    recordersref = [recorder1_ref, recorder2_ref, recorder3_ref]
    recorders0   = [recorder1_0, recorder2_0, recorder3_0]


    TimePlot(t,tmax,recorders,recordersref)
    freq, F, fftlist, fftreflist = FreqPlot(dt,c,recorders,recordersref)
    TimePlot(t,tmax,recorders0,recordersref)
    freq0, F0, fftlist0, fftreflist0 = FreqPlot(dt,c,recorders0,recordersref)

    k = np.linspace(0.1,np.pi/dx*1.1,nt)/d
    alpha = 2*np.pi - 2*np.arctan(1/4)
    u = [analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x1*dx,(y1-N)*dy]),np.array([x0*dx+2*d,2*d]),alpha = alpha),
        analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x2*dx,(y2-N)*dy]),np.array([x0*dx+2*d,2*d]),alpha = alpha),
        analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x3*dx,(y3-N)*dy]),np.array([x0*dx+2*d,2*d]),alpha = alpha)]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    axes = axes.ravel()
    for i,e in enumerate(F):
        # axes[i].plot(2*np.pi*freq/c *d,F[i], ls='--', color=f'C{i}',label = f"Numerical Zs = {int(Zs/c)}c")
        axes[i].plot(2*np.pi*freq/c *d,F0[i], ls='-', color=f'C{i}',label = r"Numerical Zs = $\infty$")
        axes[i].plot(k*d,np.abs(u[i]), ls='--', color=f'C{i}',label = r"Analytical Zs = $\infty$")
        axes[i].grid(True)
        axes[i].set_xlim(0,10)
        axes[i].set_ylim(0,2)
        axes[i].set_title(f'Recorder {i+1}')
        axes[i].legend()
    plt.show()


################################################################################################################################################
#                                                           Question 3, rectangular cells:                                                       
################################################################################################################################################



answer = input("Do you want to see our triangular wall (rectangular cell implementation)? type 'y' to continue: ")
boolean=(answer.lower() == 'y')
if boolean:

    #constants and sizes:
    c = 340 # m/s
    d = 1 # m
    kd = 10 # k = 2pi/lambda 
    if True: #Set 'True' if you wan tot pick kd
        kd = float(input(r'Give value for k_sd:'))

    dx = d/40
    dy = d/10 #roughest grid for question 1

    nx = int(7*d/dx)+int(3*d/dx)
    N = int(3/2*d/dy) # total whitespace cells for free field beneath surface
    ny = int(4*d/dy + N)+int(3*d/dy)

    Zs= 2*c
    if True: #Set 'True' if you wan tot pick Zs
        Zs = float(input('Give value for Zs/c:'))*c


    CFL = 1
    Dt = CFL/(c*np.sqrt((1/dx**2)+(1/dy**2)))
    dt = Dt/4

    #p-source
    x0 = int(d/dx)+int(3*d/dx)
    y0 = int(N + (d/10)/dy)

    A = 1
    fc= kd/(2*np.pi) * c / (d)
    a = 3
    sigma=a*d/(c*kd)
    t0 = 3*sigma
    
    #initialize fields
    ox = np.zeros((nx+1, ny))
    ox_ref = ox.copy()
    ox_0 = ox.copy()
    oy = np.zeros((nx, ny+1))
    oy_ref = oy.copy()
    oy_0 = oy.copy()
    p = np.zeros((nx, ny)) 
    p_ref = p.copy()
    p_0 = p.copy()

    # We make a mask mask_p with the same shape as p, where we select which cells belong to the triangle:
    # triangle with base d and height 2d
    base = int(2*round(d/dx/2)+1) #number of cells on the base
    height = int(round(2*d/dy)+1)

    mask_p = np.zeros((nx, ny))

    ix_start = int(x0 + d/dx)
    ix_end = ix_start + base - 1


    for j in range(N, N + height):
        mask_p[ix_start:ix_end + 1, j] = 1
        ix_start+=1
        ix_end-=1

    # generate masks for ox and oy:
    mask_ox = np.zeros((nx+1, ny))

    ix_start = int(x0 + d/dx) 
    ix_end = ix_start + base - 1 + 1

    for j in range(N, N + height):
        mask_ox[ix_start, j] = 1
        mask_ox[ix_end, j] = 1
        ix_start+=1
        ix_end-=1

    # For the impedance calculation it is useful to split the ox mask into two components:
    # a mask for the left side of the triangle and a mask for the right side
    mask_ox_left = np.zeros((nx+1, ny))

    ix_start = int(x0 + d/dx) 

    for j in range(N, N + height):
        mask_ox_left[ix_start, j] = 1
        ix_start+=1

    mask_ox_right = np.zeros((nx+1, ny))

    ix_start = int(x0 + d/dx) 
    ix_end = ix_start + base - 1 + 1

    for j in range(N, N + height):
        mask_ox_right[ix_end, j] = 1
        ix_end-=1

    mask_oy= np.zeros((nx, ny+1))

    ix_start = int(x0 + d/dx) 
    ix_end = ix_start + base - 1 

    for j in range(1,height+1):
        mask_oy[ix_start , N + j] = 1
        mask_oy[ix_end , N + j] = 1
        ix_start+=1
        ix_end-=1

    i_obj_left  = int(x0 + d/dx) 
    i_obj_right = int(i_obj_left + d/dx) 

    #recorders:
    x1 = int(i_obj_right + d/dx)
    x2 = int(x1 + d/dx)
    x3 = int(x2 + d/dx)
    y1 = int(N + d/dy/2)
    y2 = y1
    y3 = y1

    #total simulation time: when the pulse from the mirror source has travelled to the furthest mirror detector
    R = np.sqrt((3/2*d)**2 + (2*d+d/10)**2) + np.sqrt((7/2*d)**2 + (2*d+d/2)**2)
    nt = int((t0 + 5*sigma + R/c) * CFL *1.2 / Dt)
    tmax = nt

    #initialize:
    timeseries    = np.zeros((nt, 1))
    recorder1     = np.zeros((nt, 1))
    recorder1_ref = np.zeros((nt, 1))
    recorder1_0   = np.zeros((nt, 1))
    recorder2     = np.zeros((nt, 1))
    recorder2_ref = np.zeros((nt, 1))
    recorder2_0   = np.zeros((nt, 1))
    recorder3     = np.zeros((nt, 1))
    recorder3_ref = np.zeros((nt, 1))
    recorder3_0   = np.zeros((nt, 1))
   
    def updater3(c,dx,dy,dt):
        global p,ox,oy,p_ref,ox_ref,oy_ref,p_0,ox_0,oy_0,sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref
        Dt = 4*dt

        # PML:
        for _ in range(3):
            # ox has shape (nx+1, ny), oy has (nx, ny+1)
            dpx = np.append(p,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p,axis =0)
            dpy = np.append(p,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p,axis =1)
            dpx_ref = np.append(p_ref,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p_ref,axis =0)
            dpy_ref = np.append(p_ref,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p_ref,axis =1)
            dpx_0 = np.append(p_0,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p_0,axis =0)
            dpy_0 = np.append(p_0,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p_0,axis =1)

            ox_old = ox.copy() 
            oy_old = oy.copy()

            # PML:
            ox = ((1 - (sigma_x*dt)/2) * ox - dt * dpx / dx)/(1 + (sigma_x*dt)/2)
            ox_ref = ((1 - (sigma_x*dt)/2) * ox_ref - dt * dpx_ref / dx)/(1 + (sigma_x*dt)/2)
            ox_0 = ((1 - (sigma_x*dt)/2) * ox_0 - dt * dpx_0 / dx)/(1 + (sigma_x*dt)/2)

            # surface impedance:
            Ax = Zs * dt / dx
            Ay = Zs * dt / dy

            # ____Left ox components____
            idx = (mask_ox_left == 1) # we convert to boolean values

            i_idx, j_idx = np.where(idx)

            # Update ox only on the masked positions
            ox[i_idx, j_idx] = (
                (1 - Ax) * ox_old[i_idx, j_idx]
                + (2 * dt / dx) * p[i_idx - 1, j_idx]
            ) / (1 + Ax)
            ox_0[i_idx, j_idx] = 0

            # ___Right ox components___
            idx = (mask_ox_right == 1) # we convert to boolean values

            i_idx, j_idx = np.where(idx)

            # Update ox only on the masked positions
            ox[i_idx, j_idx] = (
                (1 - Ax) * ox_old[i_idx, j_idx]
                - (2 * dt / dx) * p[i_idx, j_idx]
            ) / (1 + Ax)
            ox_0[i_idx, j_idx] = 0
            
            # now we need to set p = 0 in the wall to ensure no propagation inside the wall.
            # this is done at the very end of the updater.

            dOx = ox[1:, :] - ox[:-1, :]
            dOy = oy[:, 1:] - oy[:, :-1]
            dOx_ref = ox_ref[1:, :] - ox_ref[:-1, :]
            dOy_ref = oy_ref[:, 1:] - oy_ref[:, :-1]
            dOx_0 = ox_0[1:, :] - ox_0[:-1, :]
            dOy_0 = oy_0[:, 1:] - oy_0[:, :-1]

            p = ((1 - (sigma_p * dt)/2) * p - c**2 * dt * (dOx/dx + dOy/dy))/(1 + (sigma_p * dt)/2)   
            p_ref = ((1 - (sigma_pref * dt)/2) * p_ref - c**2 * dt * (dOx_ref/dx + dOy_ref/dy))/(1 + (sigma_pref * dt)/2)
            p_0 = ((1 - (sigma_p * dt)/2) * p_0 - c**2 * dt * (dOx_0/dx + dOy_0/dy))/(1 + (sigma_p * dt)/2)

            p *= (1-mask_p)
            p_0 *= (1-mask_p)

        # ox has shape (nx+1, ny), oy has (nx, ny+1)
        dpx = np.append(p,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p,axis =0)
        dpy = np.append(p,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p,axis =1)
        dpx_ref = np.append(p_ref,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p_ref,axis =0)
        dpy_ref = np.append(p_ref,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p_ref,axis =1)
        dpx_0 = np.append(p_0,np.zeros((1,ny)),axis = 0)-np.append(np.zeros((1,ny)),p_0,axis =0)
        dpy_0 = np.append(p_0,np.zeros((nx,1)),axis = 1)-np.append(np.zeros((nx,1)),p_0,axis =1)

        ox_old = ox.copy() 
        oy_old = oy.copy()


        # PML:
        ox = ((1 - (sigma_x*dt)/2) * ox - dt * dpx / dx)/(1 + (sigma_x*dt)/2)
        ox_ref = ((1 - (sigma_x*dt)/2) * ox_ref - dt * dpx_ref / dx)/(1 + (sigma_x*dt)/2)
        ox_0 = ((1 - (sigma_x*dt)/2) * ox_0 - dt * dpx_0 / dx)/(1 + (sigma_x*dt)/2)
        
        oy = ((1 - (sigma_y*Dt)/2) * oy - Dt * dpy / dy)/(1 + (sigma_y*Dt)/2)
        oy_ref = ((1 - (sigma_yref*Dt)/2) * oy_ref - Dt * dpy_ref / dy)/(1 + (sigma_yref*Dt)/2)
        oy_0 = ((1 - (sigma_y*Dt)/2) * oy_0 - Dt * dpy_0 / dy)/(1 + (sigma_y*Dt)/2)

        # surface impedance:
        Ax = Zs * dt / dx
        Ay = Zs * dt / dy

        # ____Left ox components____
        idx = (mask_ox_left == 1) # we convert to boolean values

        i_idx, j_idx = np.where(idx)

        # Update ox only on the masked positions
        ox[i_idx, j_idx] = (
            (1 - Ax) * ox_old[i_idx, j_idx]
            + (2 * dt / dx) * p[i_idx - 1, j_idx]
        ) / (1 + Ax)
        ox_0[i_idx, j_idx] = 0

        # ___Right ox components___
        idx = (mask_ox_right == 1) # we convert to boolean values

        i_idx, j_idx = np.where(idx)

        # Update ox only on the masked positions
        ox[i_idx, j_idx] = (
            (1 - Ax) * ox_old[i_idx, j_idx]
            - (2 * dt / dx) * p[i_idx, j_idx]
        ) / (1 + Ax)
        ox_0[i_idx, j_idx] = 0

        # ___oy components___
        idy = (mask_oy == 1) # we convert to boolean values

        i_idy, j_idy = np.where(idy)

        # Update oy only on the masked positions
        oy[i_idy, j_idy] = (
            (1 - Ay) * oy_old[i_idy, j_idy]
            - (2 * Dt / dx) * p[i_idy, j_idy]
        ) / (1 + Ay)
        oy_0[i_idy, j_idy] = 0

        # now we need to set p = 0 in the wall to ensure no propagation inside the wall.
        # this is done at the very end of the updater.


        # floor: normal in y direction -> oy = 0
        oy[:, N] = 0
        oy_0[:, N] = 0

        dOx = ox[1:, :] - ox[:-1, :]
        dOy = oy[:, 1:] - oy[:, :-1]
        dOx_ref = ox_ref[1:, :] - ox_ref[:-1, :]
        dOy_ref = oy_ref[:, 1:] - oy_ref[:, :-1]
        dOx_0 = ox_0[1:, :] - ox_0[:-1, :]
        dOy_0 = oy_0[:, 1:] - oy_0[:, :-1]

        p = ((1 - (sigma_p * dt)/2) * p - c**2 * dt * (dOx/dx + dOy/dy))/(1 + (sigma_p * dt)/2)   
        p_ref = ((1 - (sigma_pref * dt)/2) * p_ref - c**2 * dt * (dOx_ref/dx + dOy_ref/dy))/(1 + (sigma_pref * dt)/2)
        p_0 = ((1 - (sigma_p * dt)/2) * p_0 - c**2 * dt * (dOx_0/dx + dOy_0/dy))/(1 + (sigma_p * dt)/2)   

        p *= (1-mask_p)
        p_0 *= (1-mask_p)


    #As the nature of the mesh cells is altered (rectangles instead of squares), a new
    #implementation of the PML function is necessary. Notice that this function is nearly 
    #entirely similar to the previously defined one. The main difference is in the thickness
    #of the PML layers, which has been reduced by a factor four for the y dimension.

    def PML_rect(D, dx , A = 10e-8, m = 4):
        #first for the velocity
        sigma_x = np.zeros_like(ox)   # shape (nx+1, ny)
        sigma_y = np.zeros_like(oy)   # shape (nx, ny+1)
        sigma_yref = np.zeros_like(oy)
        sigma_max = - (m + 1) * np.log10(A) * c / (2 * D * dx)

        for i in range(D):
            s = sigma_max * ((D-i)/D)**4
            sigma_x[i, :]      = s              
            sigma_x[-i-1, :]   = s               
            sigma_y[:, -int(i/4)-1] = s 
            sigma_yref[:, -int(i/4)-1] = s 
            sigma_yref[:,int(i/4)] = s

        sigma_px = np.zeros_like(p)   # shape (nx+1, ny)
        sigma_py = np.zeros_like(p)   # shape (nx, ny+1)
        sigma_pyref = np.zeros_like(p_ref)

        for i in range(D):
            s = sigma_max * ((D - i -1/2) / D)**4 # the -1/2 is because we work with a staggered grid
            sigma_px[i, :]      = s
            sigma_px[-i-1, :]   = s
            
            sigma_py[:, -i-1]   = s
            sigma_pyref[:, -i-1] = s 
            sigma_pyref[:,i] = s

        sigma_p = sigma_px + sigma_py
        sigma_pref = sigma_px + sigma_pyref

        return sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref

    sigma_x,sigma_y,sigma_yref,sigma_p,sigma_pref = PML_rect(dx = dx, D = int(3*d/(4*dx)))


    #---- plot of the animation ----
    fig, ax = plt.subplots()

    plt.xlim([1, nx+1])
    plt.ylim([1, 4*(ny)+1])   # <-- y-axis 4x bigger

    showplot = input("Type 'absorbing', 'free', or 'perfect' to show their respective animation:")
    showplot_i = 0*(showplot == "absorbing") + 1*(showplot == "free") + 2*(showplot == "perfect")

    movie = []

    for it in range(0, nt):
        t = (it-1)*Dt
        timeseries[it, 0] = t
        print('%d/%d' % (it, nt))

        source = A*np.cos(2*np.pi*fc*(t-t0))*np.exp(-1/2*((t-t0)**2)/(sigma**2))

        p[x0:x0+4,y0] = p[x0:x0+4,y0] + source
        p_ref[x0:x0+4,y0] = p_ref[x0:x0+4,y0] + source
        p_0[x0:x0+4,y0] = p_0[x0:x0+4,y0] + source
        updater3(c,dx,dy,dt)

        recorder1[it] = p[x1,y1]
        recorder1_ref[it] = p_ref[x1,y1]
        recorder1_0[it] = p_0[x1,y1]

        recorder2[it] = p[x2,y2]
        recorder2_ref[it] = p_ref[x2,y2]
        recorder2_0[it] = p_0[x2,y2]

        recorder3[it] = p[x3,y3]
        recorder3_ref[it] = p_ref[x3,y3]
        recorder3_0[it] = p_0[x3,y3]

        artists = [
            ax.text(0.5,1.05,'%d/%d' % (it, nt),
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes),
            ax.imshow([p.T,p_ref.T,p_0.T][showplot_i], vmin=-0.02*A, vmax=0.02*A,
                    extent=[1, nx+1, 1, 4*(ny)+1],origin='lower'),  
            
            ax.plot([int(2*d/dx)+1,int(2*d/dx)+base/2+1],[4*N,4*N+4*height], color='k',linewidth=1)[0],
            ax.plot([int(2*d/dx)+base+1,int(2*d/dx)+base/2+1],[4*N,4*N+4*height], color='k',linewidth=1)[0],
            
            ax.plot([0, nx], [4*N, 4*N], 'k-', linewidth=1)[0],  # y *4
            ax.plot(x0+2, 4*y0+2, 'ks', fillstyle="none")[0],
            ax.plot(x1, 4*y1, 'ro', fillstyle="none")[0],
            ax.plot(x2, 4*y2, 'ro', fillstyle="none")[0],
            ax.plot(x3, 4*y3, 'ro', fillstyle="none")[0],
        ]
        movie.append(artists)

    my_anim = ArtistAnimation(fig, movie, interval=50, repeat_delay=1000, blit=True)
    plt.show()


    recorders    = [recorder1, recorder2, recorder3]
    recordersref = [recorder1_ref, recorder2_ref, recorder3_ref]
    recorders0    = [recorder1_0, recorder2_0, recorder3_0]


    TimePlot(t, tmax, recorders,recordersref)
    freq, F, fftlist, fftreflist = FreqPlot(Dt,c,recorders,recordersref)
    TimePlot(t, tmax, recorders0,recordersref)
    freq0, F0, fftlist0, fftreflist0 = FreqPlot(Dt,c,recorders0,recordersref)
    
    k = np.linspace(0.1,np.pi/dx*1.1,nt)/d
    alpha = 2*np.pi - 2*np.arctan(1/4)
    u = [analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x1*dx,(y1-N)*dy]),np.array([x0*dx+2*d,2*d]),alpha = alpha),
        analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x2*dx,(y2-N)*dy]),np.array([x0*dx+2*d,2*d]),alpha = alpha),
        analytical_solution(k,np.array([x0*dx,(y0-N)*dy]),np.array([x3*dx,(y3-N)*dy]),np.array([x0*dx+2*d,2*d]),alpha = alpha)]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    axes = axes.ravel()
    for i,e in enumerate(F):
        # axes[i].plot(2*np.pi/c*freq*d,e/2,ls='--', color=f'C{i}',label = r"Numerical Zs = 2")
        axes[i].plot(2*np.pi*freq/c *d,F0[i]/2, ls='-', color=f'C{i}',label = r"Numerical Zs = $\infty$")
        axes[i].plot(k,np.abs(u[i]), ls='--', color=f'C{i}',label = r"Analytical Zs = $\infty$")
        axes[i].grid(True)
        axes[i].set_xlim(0,10)
        axes[i].set_ylim(0,3)
        axes[i].set_title(f'Recorder {i+1}')
        axes[i].legend()
    plt.show()