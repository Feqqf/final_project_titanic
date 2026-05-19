from omegaconf import OmegaConf

config = {
    '''Написать основные параметры и характеристики, которые необходимо часто менять при обучении
        А также пути к папкам и файлам используемым при работе 
    '''

    #Paths
     'paths':{
          'path_to_train': '/pythonstudy/final_project_titanic/datas/train.csv',
          'path_to_test': '/pythonstudy/final_project_titanic/datas/test.csv'
     },

    #Ids_of_columns
    'columns':{
        'target_id': 'Survived',
        'id_col': 'PassengerId'
    },
    
}