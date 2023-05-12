from setuptools import setup, find_namespace_packages

setup(name='unetr_pp',
      packages=find_namespace_packages(include=["unetr_pp", "unetr_pp.*"]),
      install_requires=[
          "torch>=1.6.0a",
          "tqdm",
          "dicom2nifti",
          "scikit-image>=0.14",
          "medpy",
          "scipy",
          "batchgenerators>=0.21",
          "numpy",
          "sklearn",
          "SimpleITK",
          "pandas",
          "requests",
          "nibabel", 'tifffile'
      ],
      entry_points={
          'console_scripts': [
              'unetr_pp_convert_decathlon_task = unetr_pp.experiment_planning.unetr_pp_convert_decathlon_task:main',
              'unetr_pp_plan_and_preprocess = unetr_pp.experiment_planning.unetr_pp_plan_and_preprocess:main',
              'unetr_pp_train = unetr_pp.run.run_training:main',
              'unetr_pp_train_DP = unetr_pp.run.run_training_DP:main',
              'unetr_pp_train_DDP = unetr_pp.run.run_training_DDP:main',
              'unetr_pp_predict = unetr_pp.inference.predict_simple:main',
              'unetr_pp_ensemble = unetr_pp.inference.ensemble_predictions:main',
              'unetr_pp_find_best_configuration = unetr_pp.evaluation.model_selection.figure_out_what_to_submit:main',
              'unetr_pp_print_available_pretrained_models = unetr_pp.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'unetr_pp_print_pretrained_model_info = unetr_pp.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'unetr_pp_download_pretrained_model = unetr_pp.inference.pretrained_models.download_pretrained_model:download_by_name',
              'unetr_pp_download_pretrained_model_by_url = unetr_pp.inference.pretrained_models.download_pretrained_model:download_by_url',
              'unetr_pp_determine_postprocessing = unetr_pp.postprocessing.consolidate_postprocessing_simple:main',
              'unetr_pp_export_model_to_zip = unetr_pp.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'unetr_pp_install_pretrained_model_from_zip = unetr_pp.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'unetr_pp_change_trainer_class = unetr_pp.inference.change_trainer:main',
              'unetr_pp_evaluate_folder = unetr_pp.evaluation.evaluator:unetr_pp_evaluate_folder',
              'unetr_pp_plot_task_pngs = unetr_pp.utilities.overlay_plots:entry_point_generate_overlay',
          ],
      },

      )