import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
import math
from matplotlib import animation
from IPython.display import HTML
from PIL import Image
import matplotlib.animation as animation
from matplotlib.patches import Circle
import streamlit.components.v1 as components  


st.set_page_config(page_title="Órbitas Relativísticas", page_icon=":comet:")

image1 = Image.open(r'C:/Users/isabe/.streamlit/titulo11.png')
st.image(image1,use_column_width='always') 
st.sidebar.image(image1,use_column_width='always')
pagina_selecionada = st.sidebar.selectbox("Selecione um tipo de órbita", ['Óbita de corpos celestes', 'Órbita de raios de luz'])


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden; }
        <style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True) 


if pagina_selecionada == "Óbita de corpos celestes":
    st.title("Corpo celeste orbitando um buraco negro")
    st.write("E se, de repente, o Sol se transformasse em um buraco negro?")
    st.write("Para isso, toda sua massa, de 2 $\cdot$ 10$^{30}$ kg (hoje espalhada numa esfera com cerca de  700.000 km de raio), deveria ser comprimida numa região com raio de cerca de  3  km.")
    image = Image.open(r'C:/Users/isabe/.streamlit/orbitaceleste.png')
    st.image(image)
    st.write("O protótipo de simulador abaixo nos permite explorar a órbita de corpos (planetas, asteroides ou espaço-naves) que se aventurassem nas vizinhanças de um buraco negro com a mesma massa do Sol.")
    st.write("Nele você pode alterar (ver figura acima):")
    st.write("* A posição inicial do corpo (em km): $x_0$")
    st.write("* O módulo da velocidade inicial do corpo: $v_0$")
    st.write("Ex.: Com $x_0$=15km e $v_0$=0.38c, obtemos uma órbita circular. No simulador abaixo, você pode testar esse e outros parâmetros.")
    st.subheader("Escolha o valor da posição inicial (em km):")   
    x0 = st.slider("Escolha entre 3km e 30km",min_value=3.0, max_value=30.0, step = 0.1)   
    st.subheader("Escolha o valor da velocidade inicial (em múltiplos da velocidade da luz):")   
    v0 = st.slider("Escolha 0.01 e 1",min_value=0.01, max_value=1.0, step = 0.01)
    result = st.button("Start")

    if result:
        def v(u, l):
            v = -u + (l ** 2) * (u ** 2) / 2 - (l ** 2) * (u ** 3)
            return v

        rs_sun = 3 # Hipótese sobre o raio de Schwarzschild do corpo central (em km). Ligeiramente maior que o do Sol.

        rst = 2.0 * (x0 / rs_sun)
        ust = 1 / rst
        l = rst * v0 
        E = v(ust, l) + 0.000000001 # Hipótese: dr/dt = 0 inicialmente.
        norbit = 10

        npoints = 500

        #print("l = ",l)
        #print("E = ",E)

        if l > math.sqrt(12):
            coef = [- 3 * (l ** 2), l ** 2, -1]
            a = np.roots(coef)
            umax = sorted(a)[1].real
            umin = sorted(a)[0].real
            vmin = v(umin, l)
            vmax = v(umax, l)

        coef = [- (l ** 2), (l ** 2) / 2, -1, -E]
        roots = np.roots(coef)
        tp1 = roots[2]
        tp2 = roots[1]
        tp3 = roots[0]

        eps = 0.00000001
        if l > math.sqrt(12):
            if E < 0 and ust < tp2.real:
                u1 = tp1.real * (1 + eps)
                u2 = tp2.real * (1 - eps)
            elif 0 < E < vmax and ust < tp2.real:
                u1 = ust / 20
                u2 = ust
                norbit = 0.5
            elif E < vmax and ust > tp3.real:
                u1 = 0.5
                u2 = tp3.real * (1 + eps)
                norbit = 0.5
            elif E > vmax:
                u1 = ust
                u2 = 0.5
                norbit = 0.5
        else:
            if E >= 0:
                u1 = ust
                u2 = 0.5
                norbit = 0.5
            else:
                u1 = tp1.real * (1 + eps)
                u2 = 0.5
                norbit = 0.5

        w = sp.Symbol('w')

        def tau_integrand(w):
            tau_integrand = w ** (-2) * (2.0 * (E - v(w, l)) ) ** (-1 / 2)
            return tau_integrand

        Ttotal, erroT = quad(tau_integrand, u1, u2) # Computes total time to go from u1 to u2.
        dt = Ttotal / npoints # Sets the time step as 1/100 of the total time.
        ud = [u2]
        for i in range(npoints+20):
            ud.append(ud[i] - dt * ud[i]**2 * (2.0 * (E - v(ud[i], l))) ** (1/2.0))
            if ud[-1].imag != 0 or math.isnan(ud[-1]):
                ud = ud[:-2]
                break
        uc = ud[::-1]
        n = len(uc)

        def theta(w):
            theta = l * (2.0 * (E - v(w, l))) ** (-1 / 2)
            return theta

        delphi, erro = quad(theta, u1, u2)

        if abs(u1 - ust) < abs(u2 - ust):
            phi1 = []
            for i in range(len(uc)):
                a = quad(theta, u1, uc[i])
                phi1.append(abs(a[0]))

            phi2 = []
            for j in range(len(ud)):
                b = quad(theta, u2, ud[j])
                phi2.append(abs(b[0]))

            if norbit == 0.5:
                utotal = uc
            else:
                utotal = np.concatenate([uc, ud] * (norbit))
        else:
            phi2 = []
            for i in range(len(uc)):
                a = quad(theta, u1, uc[i])
                phi2.append(abs(a[0]))

            phi1 = []
            for j in range(len(ud)):
                b = quad(theta, u2, ud[j])
                phi1.append(abs(b[0]))

            if norbit == 0.5:
                utotal = ud
            else:
                utotal = np.concatenate([ud, uc] * (norbit))   

        accphi = [0] * (len(utotal))

        if norbit == 0.5:
            accphi = phi1
            x = [0] * (len(utotal))
            y = [0] * (len(utotal))
            for i in range(len(utotal)):
                x[i] = (math.cos(accphi[i])) / utotal[i] * (rs_sun / 2.0)
                y[i] = (math.sin(accphi[i])) / utotal[i] * (rs_sun / 2.0)
        else:
            for i in range(norbit):
                for j in range(n):
                    accphi[j + (2 * i * n)] = 2 * i * delphi + phi1[j]
                    accphi[j + ((2 * i + 1) * n)] = ((2 * i) + 1) * delphi + phi2[j]
            x = [0] * (2 * norbit * n)
            y = [0] * (2 * norbit * n)
            for i in range(2 * norbit * n): 
                x[i] = (math.cos(accphi[i])) / utotal[i] * (rs_sun / 2.0)
                y[i] = (math.sin(accphi[i])) / utotal[i] * (rs_sun / 2.0)

        fig = plt.figure()

        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.gca().set_aspect('equal')
        ax = plt.gca()
        ax.spines['bottom'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor("black")
        circle = Circle((0, 0), rs_sun, color='dimgrey',linewidth=0)
        plt.gca().add_patch(circle)
        plt.axis([- (rs_sun / 2.0) / u1 , (rs_sun / 2.0) / u1 , - (rs_sun / 2.0) / u1 , (rs_sun / 2.0) / u1 ])
        

        # Montagem do gif

        graph, = plt.plot([], [], color="gold", markersize=3, label='Tempo: 0 s')
        L = plt.legend(loc=1)

        plt.close()  # Não mostra a imagem de fundo

        def animate(i):
            lab = 'Tempo: ' + str(round(dt*i * (rs_sun / 2.0) * 3e-5 , -int(math.floor(math.log10(abs(dt*(rs_sun / 2.0)*3e-5)))))) + ' s'
            graph.set_data(x[:i], y[:i])
            L.get_texts()[0].set_text(lab)  # Atualiza a legenda a cada frame
            return graph,

        skipframes = int(len(x)/200)
        if skipframes == 0:
            skipframes = 1

        ani1 = animation.FuncAnimation(fig, animate, frames=range(0,len(x),skipframes), interval=30, blit = True, repeat = False)
        HTML(ani1.to_jshtml())

        components.html(ani1.to_jshtml(),height=800)

    
elif pagina_selecionada == "Órbita de raios de luz":
    st.title("Luz orbitando um buraco negro")
    st.write("A teoria da relatividade geral de Einstein prevê que a trajetória da luz deve ser defletida quando passa nas vizinhanças de um corpo massivo. O protótipo de simulador abaixo nos permite explorar a órbita raios de luz ao redor de um buraco negro com a mesma massa do Sol.")
    image = Image.open(r'C:/Users/isabe/.streamlit/orbitaluz.png')
    st.image(image)
    st.write("Nele, você pode alterar (ver figura acima):")
    st.write("* O parâmetro de impacto (em km): $d$")
    st.write("Ex.: Por volta de  $d$=7.794km , temos um valor crítico para o parâmetro de impacto. No simulador abaixo, você pode testar esse e outros parâmetros.")
    st.subheader("Escolha o valor do parâmetro de impacto  d (em km):")   
    v0 = st.slider("Escolha entre 0.01 e 15",min_value=0.01, max_value=15.0, step = 0.01) #MUDAR
    result = st.button("Start")

    if result:  
        import numpy as np
        import matplotlib.pyplot as plt
        import sympy as sp
        from scipy.integrate import quad
        import math
        #from matplotlib.animation import FuncAnimation, writers

        import matplotlib
        from matplotlib import animation
        matplotlib.rc('animation', html='html5')

        from matplotlib.patches import Circle
        from IPython.display import HTML
        import warnings
        warnings.filterwarnings('ignore')
        plt.rcParams["figure.figsize"] = (6,6)

        
        d = v0 #@param {type:"slider", min:0, max:15, step:0.01} #MUDAR PARA D

        #@markdown Em seguida, clique no botão de "play", à esquerda (e aguarde alguns segundos)!

        rs_sun = 3 # Hipótese sobre o raio de Schwarzschild do corpo central (em km). Ligeiramente maior que o do Sol.
        par_imp = 2.0 * d / rs_sun #b
        k = 1/(par_imp**2)

        rst = 50
        norbit = 10

        def w(u):
            w = u**2 - 2*(u**3)
            return w

        umax = 1/3
        wmax = 1/27
        ust = 1/rst

        coef = [-2, 1 , 0, -k]
        roots = np.roots(coef)
        tp2 = roots[1]
        tp3 = roots[0]

        eps = 0.000000001

        if k < wmax and ust < umax:
            uint = ust
            uext = tp2*(1 - eps)
            norbit = 1
        elif k > wmax:
            uint = ust
            uext = 0.5 * (1 - eps)
            norbit=0.5
        elif k < wmax and ust > umax:
            uint = 0.5
            uext = tp3 * (1 + eps)
            norbit =  0.5
        else:
            print("Ha uma incoerencia entre os parametros fornecidos")

        v = sp.Symbol('v')

        def lambda_integrand(v):
            lambda_integrand = 1 / (v ** 2 ) * (k - w(v)) ** (-1 / 2)
            return lambda_integrand

        npoints = 300
        lambdatotal, errolambda = quad(lambda_integrand, uint, uext) # Computes total affine parameter to go from uint to uext.
        dlambda = lambdatotal / npoints # Sets the "time" step as 1/100 of the total "time".
        ud = [uext]
        for i in range(npoints+20):
            ud.append(ud[i] - dlambda * ud[i]**2 * (k - w(ud[i])) ** (1/2.0))
            if ud[-1].imag != 0 or math.isnan(ud[-1]):
                ud = ud[:-2]
                break
        uc = ud[::-1]
        n = len(uc)

        def theta(v):
            theta = (k-w(v))**(-1/2)
            return theta

        delphi, erro = quad(theta, uint, uext)

        phi1 = []
        for i in range(len(uc)):
            a = quad(theta, uint, uc[i])
            phi1.append(abs(a[0]))

        phi2 = []
        for j in range(len(ud)):
            b = quad(theta, uext, ud[j])
            phi2.append(abs(b[0]))

        if norbit==0.5:
            utotal = uc
        else:
            utotal = utotal = np.concatenate([uc, ud]*(norbit))

        accphi = [0]*(len(utotal))
        phi0 = np.arcsin(par_imp/rst)

        if norbit == 0.5:
            accphi = phi1
            x = [0] * (len(uc))
            y = [0] * (len(uc))
            for i in range(len(uc)):
                x[i] = (math.cos(phi0 + accphi[i])) / utotal[i] * (rs_sun / 2.0)
                y[i] = (math.sin(phi0 + accphi[i])) / utotal[i] * (rs_sun / 2.0)
        else:
            for i in range(norbit):
                for j in range(n):
                    accphi[j + (2 * i * n)] = 2 * i * delphi + phi1[j]
                    accphi[j + ((2 * i + 1) * n)] = ((2 * i) + 1) * delphi + phi2[j]
            x = [0] * (2 * norbit * n)
            y = [0] * (2 * norbit * n)
            for i in range(2 * norbit * n):
                x[i] = (math.cos(phi0 +accphi[i])) / utotal[i] * (rs_sun / 2.0)
                y[i] = (math.sin(phi0 +accphi[i])) / utotal[i] * (rs_sun / 2.0)

        fig = plt.figure()

        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.gca().set_aspect('equal')
        ax = plt.gca()
        ax.spines['bottom'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor("black")
        circle = Circle((0, 0), rs_sun, color='dimgrey')
        plt.gca().add_patch(circle)
        plt.axis([- 0.8 * (rs_sun / 2.0) / uint , 0.8 * (rs_sun / 2.0) / uint , - 0.8 * (rs_sun / 2.0) / uint , 0.8 * (rs_sun / 2.0) / uint ])

        # Montagem do gif

        graph, = plt.plot([], [], 'k--', color="gold", markersize=3)
            
        plt.close()  # Não mostra a imagem de fundo

        def animate(i):
            graph.set_data(x[:i], y[:i])
            return graph,

        skipframes = int(len(x)/200)
        if skipframes == 0:
            skipframes = 1

        ani2 = animation.FuncAnimation(fig, animate, frames=range(0,len(x),skipframes), interval=10, blit = True, repeat = False)
        HTML(ani2.to_jshtml())

        components.html(ani2.to_jshtml(),height=800)
        