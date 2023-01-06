class QPOptimizer:
    def __init__(self, calc_type: str, gamma_l: float, gamma_i: float, gamma_c: float = None) -> None:
        self._gamma_l = gamma_l
        self._gamma_i = gamma_i
        self._gamma_c = gamma_c
        self._calc_type = calc_type
    
    def __call__(self, tree_df):
        beta = tree_df.get_beta()
        I_a = tree_df.get_I_a()
        I_b = tree_df.get_I_b()
        I_c = tree_df.get_I_c()
        M = tree_df.get_mass()
        l = tree_df.calc_center_of_mass() # TODO: height only

        f_yoyo = self._gamma_i * ((I_a / I_c)**2 + (I_b / I_c)**2)
        if self._calc_type == "top":
            f = self._gamma_c * (l * M)**2 + f_yoyo
        elif self._calc_type == "yoyo":
            f = f_yoyo
            
        QP(f + self._gamma_l * 0.5 * beta.transpose() @ L @ beta)