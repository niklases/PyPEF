

import pandas as pd
from os import PathLike, path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class DatasetSplitter:
    def __init__(
            self, 
            csv_file: str | PathLike, 
            mutation_column: str | None = None, 
            separator: str | None = None, 
            n_cv: int | None = None
    ):
        self.csv_file = csv_file
        if mutation_column is None:
            mutation_column = 'mutant'
        self.mutation_column = mutation_column
        if separator is None:
            separator = ','
        self.separator = separator
        if n_cv is None:
            n_cv = 5
        self.n_cv = n_cv
        self.df = pd.read_csv(self.csv_file, sep=self.separator)
        self.random_splits_train_indices_combined, self.random_splits_test_indices_combined = None, None
        self.modulo_splits_train_indices_combined, self.modulo_splits_test_indices_combined = None, None
        self.cont_splits_train_indices_combined, self.cont_splits_test_indices_combined = None, None
        self.order_by_pos()
        self.split_random()
        self.split_modulo()
        self.split_continuous()
    
    def order_by_pos(self):
        if self.mutation_column is None:
            self.mutation_column = 'mutant'
        variants = self.df[self.mutation_column].to_list()
        self.df['variant_pos'] = [int(v[1:-1]) for v in variants]
        self.df['substitutions'] = [v[-1] for v in variants]
        self.df.sort_values(['variant_pos', 'substitutions'], ascending=[True, True], inplace=True)
        self.min_pos, self.max_pos = self.df['variant_pos'].to_numpy()[0], self.df['variant_pos'].to_numpy()[-1]
        
    def split_random(self):
        self.random_splits_train_indices_combined = []
        self.random_splits_test_indices_combined = []
        kf = KFold(n_splits=self.n_cv, shuffle=True, random_state=42)
        for i_train, i_test in kf.split(range(self.df.shape[0])):
            self.random_splits_train_indices_combined.append(i_train)
            self.random_splits_test_indices_combined.append(i_test)

    def split_modulo(self):
        """
        Likely inhomogeneous shape as not all protein backbone positions 
        are necessarily equally frequent in the dataset.
        """
        modulo_splits = [[] for _ in range(self.n_cv)]
        for i_v, v_pos in enumerate(self.df['variant_pos'].to_numpy()):
            for i in range(self.n_cv):
                if v_pos % self.n_cv == i:
                    modulo_splits[i].append(i_v)
        modulo_train_splits = []
        self.modulo_splits_train_indices_combined = []
        for i, _split in enumerate(modulo_splits):
            modulo_train_splits.append([split for j, split in enumerate(modulo_splits) if i != j])
        for splits in modulo_train_splits:
                temp = []
                for split in splits:
                    temp += split
                self.modulo_splits_train_indices_combined.append(temp)
        self.modulo_splits_test_indices_combined = [[] for _ in range(self.n_cv)]
        for i_ts, train_split in enumerate(self.modulo_splits_train_indices_combined):
            for i in range(self.df.shape[0]):
                if i not in train_split:
                    self.modulo_splits_test_indices_combined[i_ts].append(i)

    def split_continuous(self):
        """
        Similar to kf = KFold(n_splits=self.n_cv, shuffle=False) when 
        ordering variants prior to k-fold cross-validation.
        """
        cont_poses = np.array_split(np.array(range(self.min_pos, self.max_pos + 1)), self.n_cv)
        cont_splits = [[] for _ in range(self.n_cv)]
        for i_p, poses in enumerate(cont_poses):
            for i, pos in enumerate(self.df['variant_pos'].to_numpy()):
                if pos in poses:
                    cont_splits[i_p].append(i)
        cont_train_splits = []
        self.cont_splits_train_indices_combined = []
        for i, _split in enumerate(cont_splits):
            cont_train_splits.append([split for j, split in enumerate(cont_splits) if i != j])
        for splits in cont_train_splits:
                temp = []
                for split in splits:
                    temp += split
                self.cont_splits_train_indices_combined.append(temp)
        self.cont_splits_test_indices_combined = [[] for _ in range(self.n_cv)]
        for i_ts, train_split in enumerate(self.cont_splits_train_indices_combined):
            for i in range(self.df.shape[0]):
                if i not in train_split:
                    self.cont_splits_test_indices_combined[i_ts].append(i)
        
    def print_shapes(self):
        """
        Also gets inhomogeneous shapes (using for loop on sublists of nested lists instead 
        of np.shape() on entire nested list). 
        """
        random_shape_train = [np.shape(k) for k in self.random_splits_train_indices_combined]
        random_shape_test = [np.shape(k) for k in self.random_splits_test_indices_combined]
        print(f'Random train --> test split shapes: {random_shape_train} --> {random_shape_test}')
        
        modulo_shape_train = [np.shape(k) for k in self.modulo_splits_train_indices_combined]
        modulo_shape_test = [np.shape(k) for k in self.modulo_splits_test_indices_combined]
        print(f'Modulo train --> test split shapes: {modulo_shape_train} --> {modulo_shape_test}')
        
        cont_shape_train = [np.shape(k) for k in self.cont_splits_train_indices_combined]
        cont_shape_test = [np.shape(k) for k in self.cont_splits_test_indices_combined]
        print(f'Continuous train --> test split shapes: {cont_shape_train} --> {cont_shape_test}')

    def _get_zero_counts(self) -> dict:
        all_poses = np.asarray(range(self.min_pos, self.max_pos + 1))
        zero_counts = np.zeros_like(all_poses)
        return dict(zip(all_poses, zero_counts))
    
    def _get_distribution(self, indices):
        df_fold = self.df.iloc[indices, :]
        un, c = np.unique(df_fold['variant_pos'].to_numpy(), return_counts=True)
        zc = self._get_zero_counts()
        zc.update(dict(zip(un, c)))
        return list(zc.keys()), list(zc.values())
    
    def get_all_split_indices(self):
        return [
            [self.random_splits_train_indices_combined, self.random_splits_test_indices_combined],
            [self.modulo_splits_train_indices_combined, self.modulo_splits_test_indices_combined],
            [self.cont_splits_train_indices_combined, self.cont_splits_test_indices_combined]
        ]

        
    def plot_distributions(self):
        fig, axs = plt.subplots(
            nrows=4, ncols=self.n_cv, 
            figsize=((self.max_pos - self.min_pos) * 0.1 * self.n_cv, 30), 
            constrained_layout=True
        )
        poses, counts = self._get_distribution(sorted(list(self.df.index)))
        for i in range(self.n_cv):
            if i == self.n_cv // 2:
                axs[0, i].set_title("All data")
                axs[0, i].plot(poses, counts, color='black')
                axs[0, i].set_ylim(0, 20)
                axs[0, i].set_ylabel(f"# Amino acids")
            else:
                fig.delaxes(axs[0, i])
        for i_category, (train_indices, test_indices) in enumerate(self.get_all_split_indices()):
            category = ["Random", "Modulo", "Continuous"][i_category]
            for i_split in range(self.n_cv):
                pos_train, counts_train = self._get_distribution(train_indices[i_split])
                pos_test, counts_test = self._get_distribution(test_indices[i_split])
                axs[i_category + 1, i_split].plot(pos_train, counts_train)
                axs[i_category + 1, i_split].plot(pos_test, counts_test)

                xticks = list(axs[i_category + 1, i_split].get_xticks())
                if self.min_pos != 1 and not self.min_pos in xticks:
                    xticks.append(self.min_pos) 
                    xticks.append(self.max_pos)
                xticks = sorted(xticks)
                axs[i_category + 1, i_split].set_xticks(xticks)
                if i_category == 0:
                    axs[i_category + 1, i_split].set_title(f"Split {i_split + 1}")
                if i_category == 2:
                    axs[i_category + 1, i_split].set_xlabel(f"Residue position")
                if i_split == 0:
                    axs[i_category + 1, i_split].set_ylabel(f"# Amino acids")
                if i_split == self.n_cv // 2:
                    axs[i_category + 1, i_split].set_title(category)
                axs[i_category + 1, i_split].set_ylim(0, 20)
        axs[0, self.n_cv // 2].set_xticks(xticks)
        #plt.tight_layout()
        fig_path = path.abspath(path.splitext(path.basename(self.csv_file))[0] + '_pos_aa_distr.png')
        plt.savefig(fig_path, dpi=300)
        print(f"Saved figure as {fig_path}")


if __name__ == '__main__':
    d = DatasetSplitter('C:\dev\DMS_ProteinGym_substitutions\BLAT_ECOLX_Stiffler_2015.csv')
    d.print_shapes()
    d.plot_distributions()
