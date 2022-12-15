#include <iostream>
#include <mpi.h>
#include <cmath>
#include <fstream>
#include <vector>



const double	EPS = 1.e-6;                //точность
const int		nGlob = 500;               // количество разбиений
    const double	h = 1.0 / (nGlob - 1);  // шаг
const double	k = 1.0 / h;                // k^2*h^2=const
                        // JACseq,       JAC1,       JAC2,       JAC3,
                        // RED-BLACKseq, RED-BLACK1, RED-BLACK2, RED-BLACK3 :
const int flag_mass[8] = {0,0,0,1,0,0,0,1};
using namespace std;



// исходная правая часть
double fRight(double x, double y, double kGlob) {
    const double	pi = 3.1415926535;
    return 2 * sin(pi * y) + kGlob * kGlob * (1 - x) * x * sin(pi * y) + pi * pi * (1 - x) * x * sin(pi * y);
}



// исходное аналитическое решение
double fReal(double x, double y) {
    const double	pi = 3.1415926535;
    return sin(pi * y) * (1 - x) * x;
}



// норма разности векторов (новая)
double NormVectorDif(vector<double>& A, vector<double>& B, int i0, int iF) {
    double sum = 0;
    double tmp;

    for (int i = i0; i < iF; i++) {
        tmp = A[i] - B[i];
        sum += tmp * tmp;
    }
    return /*sqrt*/(sum);
}



// раскалываем матрицу на блоки  (massize - массив количества точек в каждом блоке с учетом добавочных строк)
void findSize(vector<int>& massize, int np, int id) {
    if (id == 0) {
        // заполнение массива высот без учета добавочных строк
        // если высота сетки делится нацело на количество процессов
        if( nGlob % np == 0 ) {
            for (int i = 0; i < np; i++) {
                massize[i] = (nGlob / np) * nGlob;    // (nGlob / np) - высота блока, nGlob - длина блока
            }
        } else {
            cout << "( nGlob % np != 0 )" << endl;
            // округляем (nGlob / np) и присваиваем это значение-высоту каждому процессу кроме последнего. Последний берет остаток
            int sum_of_heights_without_last = 0;
            for (int i = 0; i < np - 1; i++) {
                massize[i] = round(((double)nGlob / (double)np)) * nGlob;
                sum_of_heights_without_last += massize[i] / nGlob;
            }
            massize[np - 1] = (nGlob - sum_of_heights_without_last) * nGlob;
        }

        for (int i = 1; i < np - 1; i++) //учет добавочных строк (сверху и снизу)
            massize[i] += 2 * nGlob;
        massize[0] += nGlob;
        massize[np - 1] += nGlob;
    }

    // Один передаёт всем.
    //Параметры: что передаём, количество, тип данных, ранг процесса выполняющего рассылку, коммуникатор
    MPI_Bcast(massize.data(), np, MPI_INT, 0, MPI_COMM_WORLD);  //раздаем массив количества точек в блоках всем
}



//сбор кусочков решения в одно
void allSol(vector<int>& massize, vector<double>& sol, int np, int id) {
    vector<int>	disp(np); //массив смещений (количества точек от начала сетки до начала блока) без учета добавочных строк

    //размер принимаемого блока от каждого процесса
    int size;       //massize[] без учёта доп строк
    if ((id == 0 || id == np - 1) && np != 1) {
        size = massize[id] - nGlob;
    } else if (np != 1) {
        size = massize[id] - 2 * nGlob;
    } else {
        size = massize[id];
    }

    //считаем смещения (массив расстояний (количества точек сетки) блоков от начала сетки для каждого процесса)
    if (id == 0) {
        disp[0] = 0;
        for (int i = 1; i < np; i++)
            disp[i] = disp[i - 1] + size;
    }
    MPI_Bcast(disp.data(), np, MPI_INT, 0, MPI_COMM_WORLD);     //раздаем массив размеров смещений всем

    // именно ..v на случай неделения нацело nGlob на np (тогда size-ы будут не одинаковыми)
    //Объединение частей матрицы. Все передают одному. (Параметры: что передаём, количество, тип данных, что получаем, тип данных, ранг процесса выполняющего сбор данных, коммуникатор)
    MPI_Gatherv((id == 0) ? MPI_IN_PLACE : sol.data() + nGlob, size, MPI_DOUBLE, sol.data(), massize.data(), disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

}



// вычисление аналитического решения на сетке
void fToGridReal(vector<double>& solReal, double step) {
    for (int i = 0; i < nGlob; i++)
        for (int j = 0; j < nGlob; j++)
            solReal[i * nGlob + j] = fReal(i * step, j * step);
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//метод Якоби последовательный
double Jacobi_seq(vector<double>& sol, vector<double>& solPrev, int& iter) {
    double normFinal;
    double coeff = (4. + h * h * k * k);

    iter = 0;
    do {
        iter++;
        for (int i = 1; i < nGlob - 1; i++) // метод Якоби
            for (int j = 1; j < nGlob - 1; j++)
                sol[i * nGlob + j] = (h * h * fRight(i * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;

        normFinal = sqrt(NormVectorDif(sol, solPrev, 0, nGlob * nGlob));

        solPrev.swap(sol);
    } while (normFinal > EPS);

    return normFinal;
}



// процедура передачи строк между процессами (используется в RedBlack_3 и во всех Jacobi_PAR)
void func_transfer(vector<double>& solPrev, vector<int>& massize, int id, int np, int iter, MPI_Request* reqSENDsAB, MPI_Request* reqRESVsAB, MPI_Request* reqSENDsBA, MPI_Request* reqRESVsBA, int flag_meth) {
    switch (flag_meth) {
        // JACOBI_1 (Send + Recv)
        case 1: {
            // принимаем доп строки, отправляем крайние НЕдоп строки
            // 1 отправка вниз (волна вниз) (первый параметр - отступаем от верха блока (от начала верхней доп. строки) столько чтобы попасть на начало нижней НЕдоп. строки)
            MPI_Send(solPrev.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE, (id != np - 1) ? id + 1 : 0, 56, MPI_COMM_WORLD);
            // 2 принятие сверху (волна вниз) (первый параметр - принимаем строку сверху поэтому кладём её в самый верх нашего блока, т.е. в верхную доп строку)
            MPI_Recv(solPrev.data(), nGlob, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            // 3 отправка вверх (волна вверх) (первый параметр - отступаем от начала верхней доп строки одну строку чтобы попасть (и передать) на первую НЕдоп строку)
            MPI_Send(solPrev.data() + nGlob, (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 65, MPI_COMM_WORLD);
            // 4 принятие снизу (волна вверх) (первый параметр - отступаем от начала верхней доп строки стольуо чтобы попасть на начало нижней доп строки)
            MPI_Recv(solPrev.data() + massize[id] - nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE, (id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        }
            break;
        // JACOBI_2 (Sendrecv)
        case 2: {
            // 1 и 2 (волна пересылок вниз, тег всегда 56) отправить вниз, а принять сверху
            MPI_Sendrecv(solPrev.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE, (id != np - 1) ? id + 1 : 0, 56, solPrev.data(), (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            // 3 и 4(волна пересылок вверх, тег всегда 65) отправляем вверх, принимаем снизу
            MPI_Sendrecv(solPrev.data() + nGlob, (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 65, solPrev.data() + massize[id] - nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE, (id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        }
            break;
        // JACOBI_3 & RED-BLACK_3 (Isend + Irecv)
        case 3: {
            if (iter % 2 == 0) {
                // Ч Ë Т   I T E R
                MPI_Startall(2, reqSENDsAB);
                MPI_Startall(2, reqRESVsAB);
            }
            else {
                // Н Е Ч Ë Т   I T E R
                MPI_Startall(2, reqSENDsBA);
                MPI_Startall(2, reqRESVsBA);
            }

        }
            break;
        default:
            cout << "ERROR CASE in func_transfer()\n";
    }
}



// общая функция для параллельных методов Якоби
double Jacobi_PAR(vector<double>& sol, vector<double>& solPrev, vector<int>& massize, int id, int np, int& iter, int flag_meth) {
    double locnorm, normFinal; //locnorm -- локальная погрешность блока, данного каждому процессору для вычисления нормы ошибки
    double coeff = (4. + h * h * k * k);

    // сдвиг по высоте сетки для каждого процесса, то есть начало верхней доп. строки для каждого блока
    int shift = 0; // равно нулю для нулевого процессора

    if(id == np - 1) {
        shift = nGlob - (massize[id] / nGlob); // поднимаемся с конца на высоту последнего блока с учётом доп строки
    } else if(id != 0) {
        shift = id * (massize[id] / nGlob - 2) - 1; //спускаемся с начала на id*(высота среднего блока без учета доп строк) и поднимаемся на одну доп строку
    }

    // нужно только для Isend + Irecv
    MPI_Request* reqSENDsAB = new MPI_Request[2];
    MPI_Request* reqRESVsAB = new MPI_Request[2];
    MPI_Request* reqSENDsBA = new MPI_Request[2];
    MPI_Request* reqRESVsBA = new MPI_Request[2];

    // И Н И Ц И А Л И З А Ц И Я   I s e n d   нужно только для Isend + Irecv
    if (flag_meth == 3) {
        // Ч Ë Т   I T E R
        // 1 Отправляем сверху вниз AB
        MPI_Send_init(solPrev.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,(id != np - 1) ? id + 1 : 0, 56, MPI_COMM_WORLD, reqSENDsAB);
        // 4 Получаем снизу вверх AB
        MPI_Recv_init(solPrev.data() + massize[id] - nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,(id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, reqRESVsAB);
        // 3 Отправляем снизу вверх AB
        MPI_Send_init(solPrev.data() + nGlob, (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 65,MPI_COMM_WORLD, reqSENDsAB + 1);
        // 2 Получаем сверху вниз AB
        MPI_Recv_init(solPrev.data(), (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56,MPI_COMM_WORLD, reqRESVsAB + 1);

        // Н Е Ч Ë Т   I T E R
        // 1 Отправляем сверху вниз BA
        MPI_Send_init(sol.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,(id != np - 1) ? id + 1 : 0, 56, MPI_COMM_WORLD, reqSENDsBA);
        // 4 Получаем снизу вверх BA
        MPI_Recv_init(sol.data() + massize[id] - nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,(id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, reqRESVsBA);
        // 3 Отправляем снизу вверх BA
        MPI_Send_init(sol.data() + nGlob, (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 65,MPI_COMM_WORLD, reqSENDsBA + 1);
        // 2 Получаем сверху вниз BA
        MPI_Recv_init(sol.data(), (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56, MPI_COMM_WORLD,reqRESVsBA + 1);
    }

    iter = 0;
    do {
        func_transfer(solPrev, massize, id, np, iter, reqSENDsAB, reqRESVsAB, reqSENDsBA, reqRESVsBA, flag_meth);

        //центральная часть (без первой и последней НЕдоп. строки)
        for (int i = 2; i < massize[id] / nGlob - 2; i++)
            for (int j = 1; j < nGlob - 1; j++)
                sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;

        // опять же только для Isend + Irecv
        if (flag_meth == 3) {
                MPI_Waitall(2, (iter % 2 == 0) ? reqSENDsAB : reqSENDsBA, MPI_STATUSES_IGNORE);
                MPI_Waitall(2, (iter % 2 == 0) ? reqRESVsAB : reqRESVsBA, MPI_STATUSES_IGNORE);
        }

        // начало блока (расчет первой НЕдоп. строки блока)
        int i = 1;
        for (int j = 1; j < nGlob - 1; j++)
            sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;

        //конец блока (расчет последней НЕдоп. строки блока)
        i = massize[id] / nGlob - 2;
        for (int j = 1; j < nGlob - 1; j++)
            sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;

        locnorm = NormVectorDif(sol, solPrev, (id == 0) ? 0 : nGlob,(id == np-1) ? massize[id] : massize[id] - nGlob);
        // объединяет значения из всех процессов и распределяет результат обратно во все процессы.
        // что отправляем, что получает, количество , тип, ОПРЕДЕЛЕНИЕ МАКСИМАЛЬНОГО ЗНАЧЕНИЯ, коммуникатор
        MPI_Allreduce(&locnorm, &normFinal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        normFinal = sqrt(normFinal);

        solPrev.swap(sol);

        iter++;

        //cout << iter << ": " << normFinal << endl;

    } while (normFinal > EPS);

    return normFinal;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



//метод красно-черных итераций последовательный
double RedBlack_seq(vector<double>& sol, vector<double>& solPrev, int& iter) {
    double normFinal;

    double coeff = (4. + h * h * k * k);

    iter = 0;
    do {
        for (int i = 1; i < nGlob - 1; i++) // метод красно-черных
            for (int j = (i % 2) + 1; j < nGlob - 1; j += 2)

                sol[i * nGlob + j] = (h * h * fRight(i * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;

        for (int i = 1; i < nGlob - 1; i++) // метод красно-черных
            for (int j = ((i + 1) % 2) + 1; j < nGlob - 1; j += 2)

                sol[i * nGlob + j] = (h * h * fRight(i * h, j * h, k) + (sol[i * nGlob + j - 1] + sol[i * nGlob + j + 1] + sol[(i - 1) * nGlob + j] + sol[(i + 1) * nGlob + j])) / coeff;

        normFinal = sqrt(NormVectorDif(sol, solPrev, 0, nGlob * nGlob));

        solPrev.swap(sol);

        iter++;

    } while (normFinal > EPS);

    return normFinal;
}



//метод красно-черных send+recv
double RedBlack_1(vector<double>& sol, vector<double>& solPrev, vector<int> massize, int id, int np, int& iter) {
    double locnorm, normFinal = 0;

    double coeff = (4. + h * h * k * k);

    // сдвиг по высоте сетки для каждого процесса, то есть начало верхней доп. строки для каждого блока
    int shift = 0; // равно нулю для нулевого процессора

    if (id == np - 1) {
        shift = nGlob - (massize[id] / nGlob); // поднимаемся с конца на высоту последнего блока с учётом доп строки
    } else if (id != 0){
        shift = id * (massize[id] / nGlob - 2) - 1; //спускаемся с начала на id*(высота среднего блока без учета доп строк) и поднимаемся на одну доп строку
    }

    iter = 0;
    do {
        // 1
        MPI_Send(solPrev.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,   (id != np - 1) ? id + 1 : 0, 56, MPI_COMM_WORLD);
        // 2
        MPI_Recv(solPrev.data(),                           (id != 0) ? nGlob : 0,      MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        // 3
        MPI_Send(solPrev.data() + nGlob,                   (id != 0) ? nGlob : 0,      MPI_DOUBLE,   (id != 0) ? id - 1 : np - 1, 65, MPI_COMM_WORLD);
        // 4
        MPI_Recv(solPrev.data() + massize[id] - nGlob,     (id != np - 1) ? nGlob : 0, MPI_DOUBLE, (id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        for (int i = 1; i < massize[id] / nGlob - 1; i++) // метод красно-черных
            for (int j = ((i + shift) % 2) + 1; j < nGlob - 1; j += 2) {
                sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;
            }

        // 1
        MPI_Send(sol.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,   (id != np - 1) ? id + 1 : 0, 56, MPI_COMM_WORLD);//0
        // 2
        MPI_Recv(sol.data(),                           (id != 0) ? nGlob : 0,      MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);//2
        // 3
        MPI_Send(sol.data() + nGlob,                   (id != 0) ? nGlob : 0,      MPI_DOUBLE,   (id != 0) ? id - 1 : np - 1, 65, MPI_COMM_WORLD);//1
        // 4
        MPI_Recv(sol.data() + massize[id] - nGlob,     (id != np - 1) ? nGlob : 0, MPI_DOUBLE, (id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


        for (int i = 1; i < massize[id] / nGlob - 1; i++) // метод красно-черных
            for (int j = (((i + shift) + 1) % 2) + 1; j < nGlob - 1; j += 2) {
                sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (sol[i * nGlob + j - 1] + sol[i * nGlob + j + 1] + sol[(i - 1) * nGlob + j] + sol[(i + 1) * nGlob + j])) / coeff;
            }

        locnorm = NormVectorDif(sol, solPrev, (id == 0) ? 0 : nGlob, (id == np) ? massize[id] : massize[id] - nGlob);
        MPI_Allreduce(&locnorm, &normFinal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        normFinal = sqrt(normFinal);

        solPrev.swap(sol);

        iter++;

    } while (normFinal > EPS);

    return normFinal;
}



//метод красно-черных sendrecv
double RedBlack_2(vector<double>& sol, vector<double>& solPrev, vector<int> massize, int id, int np, int& iter) {
    double locnorm, normFinal;

    double coeff = (4. + h * h * k * k);

    // сдвиг по высоте сетки для каждого процесса, то есть начало верхней доп. строки для каждого блока
    int shift = 0; // равно нулю для нулевого процессора

    if (id == np - 1) {
        shift = nGlob - (massize[id] / nGlob); // поднимаемся с конца на высоту последнего блока с учётом доп строки
    } else if (id != 0){
        shift = id * (massize[id] / nGlob - 2) - 1; //спускаемся с начала на id*(высота среднего блока без учета доп строк) и поднимаемся на одну доп строку
    }

    iter = 0;
    do {
        MPI_Sendrecv(solPrev.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,  (id != np - 1) ? id + 1 : 0, 56, \
                     solPrev.data(),                            (id != 0) ? nGlob : 0,      MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Sendrecv(solPrev.data() + nGlob,                   (id != 0) ? nGlob : 0,      MPI_DOUBLE,  (id != 0) ? id - 1 : np - 1, 65, \
                     solPrev.data() + massize[id] - nGlob,      (id != np - 1) ? nGlob : 0, MPI_DOUBLE, (id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        for (int i = 1; i < massize[id] / nGlob - 1; i++) // метод красно-черных
            for (int j = ((i + shift) % 2) + 1; j < nGlob - 1; j += 2)
                sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;

        MPI_Sendrecv(sol.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,  (id != np - 1) ? id + 1 : 0, 56,\
                     sol.data(),                            (id != 0) ? nGlob : 0,      MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Sendrecv(sol.data() + nGlob,                   (id != 0) ? nGlob : 0,      MPI_DOUBLE,  (id != 0) ? id - 1 : np - 1, 65, \
                     sol.data() + massize[id] - nGlob,      (id != np - 1) ? nGlob : 0, MPI_DOUBLE, (id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        for (int i = 1; i < massize[id] / nGlob - 1; i++) // метод красно-черных
            for (int j = (((i + shift) + 1) % 2) + 1; j < nGlob - 1; j += 2)
                sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (sol[i * nGlob + j - 1] + sol[i * nGlob + j + 1] + sol[(i - 1) * nGlob + j] + sol[(i + 1) * nGlob + j])) / coeff;

        locnorm = NormVectorDif(sol, solPrev, (id == 0) ? 0 : nGlob, (id == np) ? massize[id] : massize[id] - nGlob);
        MPI_Allreduce(&locnorm, &normFinal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        normFinal = sqrt(normFinal);

        solPrev.swap(sol);

        iter++;

    } while (normFinal > EPS);

    return normFinal;
}



//метод красно-черных isend+irecv
double RedBlack_3(vector<double>& sol, vector<double>& solPrev, vector<int> massize, int id, int np, int& iter) {
    double locnorm, normFinal;

    double coeff = (4. + h * h * k * k);

    // сдвиг по высоте сетки для каждого процесса, то есть начало верхней доп. строки для каждого блока
    int shift = 0; // равно нулю для нулевого процессора

    if (id == np - 1) {
        shift = nGlob - (massize[id] / nGlob); // поднимаемся с конца на высоту последнего блока с учётом доп строки
    } else if (id != 0){
        shift = id * (massize[id] / nGlob - 2) - 1; //спускаемся с начала на id*(высота среднего блока без учета доп строк) и поднимаемся на одну доп строку
    }

    // нужно только для Isend + Irecv
    MPI_Request* reqSENDsAB = new MPI_Request[2];
    MPI_Request* reqRESVsAB = new MPI_Request[2];
    MPI_Request* reqSENDsBA = new MPI_Request[2];
    MPI_Request* reqRESVsBA = new MPI_Request[2];

    // И Н И Ц И А Л И З А Ц И Я   I s e n d   нужно только для Isend + Irecv

    // Ч Ë Т   I T E R
    // 1 Отправляем сверху вниз AB
    MPI_Send_init(solPrev.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,(id != np - 1) ? id + 1 : 0, 56, MPI_COMM_WORLD, reqSENDsAB);
    // 4 Получаем снизу вверх AB
    MPI_Recv_init(solPrev.data() + massize[id] - nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,(id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, reqRESVsAB);
    // 3 Отправляем снизу вверх AB
    MPI_Send_init(solPrev.data() + nGlob, (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 65,MPI_COMM_WORLD, reqSENDsAB + 1);
    // 2 Получаем сверху вниз AB
    MPI_Recv_init(solPrev.data(), (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56,MPI_COMM_WORLD, reqRESVsAB + 1);

    // Н Е Ч Ë Т   I T E R
    // 1 Отправляем сверху вниз BA
    MPI_Send_init(sol.data() + massize[id] - 2 * nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,(id != np - 1) ? id + 1 : 0, 56, MPI_COMM_WORLD, reqSENDsBA);
    // 4 Получаем снизу вверх BA
    MPI_Recv_init(sol.data() + massize[id] - nGlob, (id != np - 1) ? nGlob : 0, MPI_DOUBLE,(id != np - 1) ? id + 1 : 0, 65, MPI_COMM_WORLD, reqRESVsBA);
    // 3 Отправляем снизу вверх BA
    MPI_Send_init(sol.data() + nGlob, (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 65,MPI_COMM_WORLD, reqSENDsBA + 1);
    // 2 Получаем сверху вниз BA
    MPI_Recv_init(sol.data(), (id != 0) ? nGlob : 0, MPI_DOUBLE, (id != 0) ? id - 1 : np - 1, 56, MPI_COMM_WORLD,reqRESVsBA + 1);


    iter = 0;
    do {
        func_transfer(solPrev, massize, id, np, iter /*Prev поэтому AB нужно start-ануть*/, reqSENDsAB, reqRESVsAB, reqSENDsBA, reqRESVsBA, 3);

        //центральная часть
        for (int i = 2; i < massize[id] / nGlob - 2; i++) // метод Красно-черных
            for (int j = ((i + shift) % 2) + 1; j < nGlob - 1; j += 2)
                sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;

        // опять же для Isend + Irecv
        MPI_Waitall(2, (iter % 2 == 0) ? reqSENDsAB : reqSENDsBA /*Prev поэтому AB*/, MPI_STATUSES_IGNORE);
        MPI_Waitall(2, (iter % 2 == 0) ? reqRESVsAB : reqRESVsBA /*Prev поэтому AB*/, MPI_STATUSES_IGNORE);


        //начало
        int i = 1;
        for (int j = ((i + shift) % 2) + 1; j < nGlob - 1; j++)
            sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;

        //конец
        i = massize[id] / nGlob - 2;
        for (int j = ((i + shift) % 2) + 1; j < nGlob - 1; j++)
            sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (solPrev[i * nGlob + j - 1] + solPrev[i * nGlob + j + 1] + solPrev[(i - 1) * nGlob + j] + solPrev[(i + 1) * nGlob + j])) / coeff;



        func_transfer(solPrev, massize, id, np, iter+1 /*sol поэтому BA нужно start-ануть*/, reqSENDsAB, reqRESVsAB, reqSENDsBA, reqRESVsBA, 3);

        //центральная часть
        for (int i = 2; i < massize[id] / nGlob - 2; i++) // метод красно-черный
            for (int j = (((i + shift) + 1) % 2) + 1; j < nGlob - 1; j += 2)
                sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (sol[i * nGlob + j - 1] + sol[i * nGlob + j + 1] + sol[(i - 1) * nGlob + j] + sol[(i + 1) * nGlob + j])) / coeff;

        // опять же для Isend + Irecv
        MPI_Waitall(2, (iter % 2 == 0) ? reqSENDsBA : reqSENDsAB /*sol поэтому BA*/, MPI_STATUSES_IGNORE);
        MPI_Waitall(2, (iter % 2 == 0) ? reqRESVsBA : reqRESVsAB /*sol поэтому BA*/, MPI_STATUSES_IGNORE);


        //начало
        i = 1;
        for (int j = (((i + shift) + 1) % 2) + 1; j < nGlob - 1; j += 2)
            sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (sol[i * nGlob + j - 1] + sol[i * nGlob + j + 1] + sol[(i - 1) * nGlob + j] + sol[(i + 1) * nGlob + j])) / coeff;

        //конец
        i = massize[id] / nGlob - 2;
        for (int j = (((i + shift) + 1) % 2) + 1; j < nGlob - 1; j += 2)
            sol[i * nGlob + j] = (h * h * fRight((i + shift) * h, j * h, k) + (sol[i * nGlob + j - 1] + sol[i * nGlob + j + 1] + sol[(i - 1) * nGlob + j] + sol[(i + 1) * nGlob + j])) / coeff;



        locnorm = NormVectorDif(sol, solPrev, (id == 0) ? 0 : nGlob, (id == np) ? massize[id] : massize[id] - nGlob);
        MPI_Allreduce(&locnorm, &normFinal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        normFinal = sqrt(normFinal);

        solPrev.swap(sol);

        iter++;

    } while (normFinal > EPS);

    return normFinal;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



void method_COUT(double T, double SPEEDUP, int iter, double normFinal, double errorFinal) {
    if (SPEEDUP != 99)
        cout << "      SPEEDUP = " << SPEEDUP << " <-------" << endl;
    cout << "         time = " << T << endl;
    cout << "         iter = " << iter << endl;
    cout << "  ||k+1 - k|| = " << normFinal << endl;
    cout << "        error = " << errorFinal << endl;
}



void method_RUN (int id, int np, vector<int> &massize, double &Tseq, int flag_meth) {
    double T, normFinal, errorFinal;;
    int iter;

    vector<double> sol, solPrev, solReal;

    if (id == 0) {
        solReal.resize(nGlob * nGlob);
        //матрица реальных значений
        fToGridReal(solReal, h);
    }

    if(id != 0) {
        sol.resize(massize[id], 0);
        solPrev.resize(massize[id], 0);
    } else {
        sol.resize(nGlob * nGlob, 0);
        solPrev.resize(nGlob * nGlob, 0);
    }

    sol.assign(sol.size(), 0);
    solPrev.assign(solPrev.size(), 0);

    switch (flag_meth) {
        case 0: {
            if(id == 0) {
                Tseq = -MPI_Wtime();
                normFinal = Jacobi_seq(sol, solPrev, iter);
                Tseq += MPI_Wtime();

                errorFinal = NormVectorDif(sol, solReal, 0, nGlob * nGlob);
                cout << "--------------------------------\nJACOBI seq:" << endl;
                // 99 чтобы спидап не выводился для данного последовательного метода
                method_COUT(Tseq, 99, iter, normFinal, errorFinal);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
            break;

        case 1: {
            if(id == 0)
                T = -MPI_Wtime();
            normFinal = Jacobi_PAR(sol, solPrev, massize, id, np, iter, 1);
            if(id == 0) {
                T += MPI_Wtime();
            }

            //сбор кусочков решения в одно
            allSol(massize, sol, np, id);

            if(id == 0) {
                errorFinal = NormVectorDif(sol, solReal, 0, nGlob * nGlob);
                cout << "JACOBI 1: \t\t\t(Send + Recv)" << endl;
                // (flag_mass[0]) ? (Tseq / T) : 0  --  считаем обычный спидап если JACseq вообще считается до этого и считаем спидап нулевым иначе
                method_COUT(T, (flag_mass[0]) ? (Tseq / T) : 0, iter, normFinal, errorFinal);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
            break;

        case 2: {
            if(id == 0)
                T = -MPI_Wtime();
            normFinal = Jacobi_PAR(sol, solPrev, massize, id, np, iter, 2);
            if(id == 0)
                T += MPI_Wtime();

            //сбор кусочков решения в одно
            allSol(massize, sol, np, id);

            if(id == 0) {
                errorFinal = NormVectorDif(sol, solReal, 0, nGlob * nGlob);
                cout << "JACOBI 2: \t\t\t(Sendrecv)" << endl;
                // (flag_mass[0]) ? (Tseq / T) : 0  --  считаем обычный спидап если JACseq вообще считается до этого и считаем спидап нулевым иначе
                method_COUT(T, (flag_mass[0]) ? (Tseq / T) : 0, iter, normFinal, errorFinal);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
            break;

        case 3: {
            if(id == 0)
                T = -MPI_Wtime();
            normFinal = Jacobi_PAR(sol, solPrev, massize, id, np, iter, 3);
            if(id == 0)
                T += MPI_Wtime();

            //сбор кусочков решения в одно
            allSol(massize, sol, np, id);

            if(id == 0) {
                errorFinal = NormVectorDif(sol, solReal, 0, nGlob * nGlob);
                cout << "JACOBI 3: \t\t\t(Isend + Iresv)" << endl;
                // (flag_mass[0]) ? (Tseq / T) : 0  --  считаем обычный спидап если JACseq вообще считается до этого и считаем спидап нулевым иначе
                method_COUT(T, (flag_mass[0]) ? (Tseq / T) : 0, iter, normFinal, errorFinal);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
            break;

        case 4: {
            if (id == 0) {
                Tseq = -MPI_Wtime();
                normFinal = RedBlack_seq(sol, solPrev, iter);
                Tseq += MPI_Wtime();

                errorFinal = NormVectorDif(sol, solReal, 0, nGlob * nGlob);
                cout << "--------------------------------\nRED-BLACK seq:" << endl;
                // 99 чтобы спидап не выводился для данного последовательного метода
                method_COUT(Tseq, 99, iter, normFinal, errorFinal);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
            break;

        case 5: {
            if (id == 0)
                T = -MPI_Wtime();
            normFinal = RedBlack_1(sol, solPrev, massize, id, np, iter);
            if (id == 0)
                T += MPI_Wtime();

            //сбор кусочков решения в одно
            allSol(massize, sol, np, id);

            if (id == 0) {
                errorFinal = NormVectorDif(sol, solReal, 0, nGlob * nGlob);
                cout << "RED-BLACK 1:      \t\t(Send + Recv)" << endl;
                // (flag_mass[4]) ? (Tseq / T) : 0  --  считаем обычный спидап если REDseq вообще считается до этого и считаем спидап нулевым иначе
                method_COUT(T, (flag_mass[4]) ? (Tseq / T) : 0, iter, normFinal, errorFinal);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
            break;

        case 6: {
            if (id == 0)
                T = -MPI_Wtime();
            normFinal = RedBlack_2(sol, solPrev, massize, id, np, iter);
            if (id == 0)
                T += MPI_Wtime();

            //сбор кусочков решения в одно
            allSol(massize, sol, np, id);

            if (id == 0) {
                errorFinal = NormVectorDif(sol, solReal, 0, nGlob * nGlob);
                cout << "RED-BLACK 2:      \t\t(Sendrecv)" << endl;
                // (flag_mass[4]) ? (Tseq / T) : 0  --  считаем обычный спидап если REDseq вообще считается до этого и считаем спидап нулевым иначе
                method_COUT(T, (flag_mass[4]) ? (Tseq / T) : 0, iter, normFinal, errorFinal);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
            break;

        case 7: {
            if (id == 0)
                T = -MPI_Wtime();
            normFinal = RedBlack_3(sol, solPrev, massize, id, np, iter);
            if (id == 0)
                T += MPI_Wtime();

            //сбор кусочков решения в одно
            allSol(massize, sol, np, id);

            if (id == 0) {
                errorFinal = NormVectorDif(sol, solReal, 0, nGlob * nGlob);
                cout << "RED-BLACK 3:      \t\t(Isend + Iresv)" << endl;
                // (flag_mass[4]) ? (Tseq / T) : 0  --  считаем обычный спидап если REDseq вообще считается до этого и считаем спидап нулевым иначе
                method_COUT(T, (flag_mass[4]) ? (Tseq / T) : 0, iter, normFinal, errorFinal);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
            break;

        default:
            cout << "ERROR CASE in method_RUN()\n";
    }

    sol.assign(sol.size(), 0);
    solPrev.assign(solPrev.size(), 0);
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



int main(int argc, char** argv)
{
    int id, np;
    double Tseq;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);


    vector<int>	massize(np);

    //распределение размеров блоков
    findSize(massize, np, id);

    if (id == 0) {
        cout << " Size = " << nGlob << endl;
        cout << "  EPS = " << EPS << endl;
        cout << "   np = " << np << endl;
        //вывод высот блоков:
        cout << "    heights: " << massize[0] / nGlob - 1 << " (excluding additional lines)" << endl;
        if (np <= 5) {
            for (int j = 1; j < np - 1; ++j) {
                cout << "             " << massize[j] / nGlob - 2 << endl;
            }
            cout << "             " << massize[np - 1] / nGlob - 1 << endl;
        } else {
            cout << "             " << massize[1] / nGlob - 2 << endl;
            cout << "             ..." << endl;
            cout << "             " << massize[np - 2] / nGlob - 2 << endl;
            cout << "             " << massize[np - 1] / nGlob - 1 << endl;
        }
    }

    //идём по массиву флагов flag_mass и для каждого флага вызываем соответствующий метод если там стоит 1
    for (int j = 0; j < 8; ++j) {
        if (flag_mass[j]) {
            method_RUN(id, np, massize, Tseq, j);
        }
    }

    if (id == 0) {



        cout << "\n\n\n\n\n";
    }

    MPI_Finalize();
}