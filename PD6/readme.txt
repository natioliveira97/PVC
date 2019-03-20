# PD6
Nome: Lívia Gomes Costa Fonseca
Matrícula: 16/0034078

Nome: Natalia Oliveira Borges
Matrícula: 16/0015863

Esse projeto foi feito em python3.5 e testado em máquinas Linux Xubuntu versão 18.04 e em Linux Ubuntu versão 16.04, ambos usando Opencv versão 3.2.0 e numpy versão 3.5.


Link para o trabalho no github: https://github.com/Liviagcf/PD6 (nesse link baixe a pasta data para ter acesso ao banco de imagens utilizado e videos, mandamos um video para teste nesse arquivo)



###### Gerando imagens de treinamento ###########
Dentro da pasta src faça:

python 3.5 create_imagedata.py nome_do_video (sem extensão)

O video irá passar de frame em frame e em cada frame deve-se selecionar uma região e classificar como bola 'p' ou não bola 'n'. Para passar o frame é só clicar qualquer outra tecla.

Os videos devem estar dentro da pasta 'data'



###### Gerando gabarito ###########
Dentro da pasta src faça:

python 3.5 gab_generate.py nome_do_video (sem extensão)

O video irá passar de frame em frame e em cada frame deve-se clicar na bola e apertar 'p' ou apertar qualquer tecla para framens sem bola.

Os videos devem estar dentro da pasta 'data'


###### Treinando o método ###########
Dentro da pasta src faça:

opencv_createsamples -info positive.info -num NUMERO_DE_IMAGENS_POSITIVAS -w 20 -h 20 -vec ball.vec

opencv_traincascade -data train -vec ball.vec -bg negative.txt -numPos NUMERO_DE_IMAGENS_POSITIVAS -numNeg NUMERO_DE_IMAGENS_NEGATIVAS -numStages ITERACOES -w 20 -h 20 -featureType METODO -acceptanceRatioBreakValue 0.00001 -maxFalseAlarmRate 0.2

Em METODO temos duas opções: HAAR ou LBP

###### Testando o método #######
Dentro da pasta src faça:

python 3.5 pd6.py nome_do_video (sem extensão)


# PD6
