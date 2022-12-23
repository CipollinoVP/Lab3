#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>

const double eps = 1.e-5;
const int N = 1260; //количество точек
const double h = 1.0 / (N - 1); //шаг
const double k = 1.0 / h;
const double coef = (4.0 + h * h * k * k);



//заполняем вектор нулями
void zero(std::vector<double>& A)
{
    for (double & i : A)
        i = 0.0;
}


//правая часть
double	f(double x, double y)
{
    return 2 * sin(M_PI * y) + k * k * (1 - x) * x * sin(M_PI * y) + M_PI * M_PI * (1 - x) * x * sin(M_PI * y);
}


double norm(std::vector<double>& A, std::vector<double>& B, int i_beg, int i_end)
{
    double norma = 0.0;
    for (int i = i_beg; i < i_end; ++i)
        if (norma < fabs(A[i] - B[i]))
            norma = fabs(A[i] - B[i]);
    return norma;
}


void general_y(std::vector<int>& el_num, std::vector<double>& y_n, std::vector<double>& y, std::vector<int>& displs, int np, int myid)
{
    //реальный размер блока от каждого процесса
    int size;
    if ((myid == 0 || myid == np - 1) && np != 1)
        size = el_num[myid] - N;
    else if (np != 1)
        size = el_num[myid] - 2 * N;
    else
        size = el_num[myid];

    //объединение частей матрицы
    MPI_Gatherv((myid == 0) ? y_n.data() : y_n.data() + N, size, MPI_DOUBLE, y.data(), el_num.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void analyt_sol(std::vector<double>& u)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            u[i * N + j] = (1 - i * h) * i * h * sin(M_PI * j * h);
}

double Jacobi(std::vector<double>& y, std::vector<double>& y_n, std::vector<int>& el_num, int myid, int np, int& iterations, int send_type)
{
    double norm_f;
    //последовательный
    if (np == 1)
    {

        iterations = 0;
        do
        {
            ++iterations;
            for (int i = 1; i < N - 1; ++i)
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f(i * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;

            norm_f = norm(y, y_n, 0, N * N);
            y_n.swap(y);
        } while (norm_f > eps);
    }
    //параллельный
    if (np > 1)
    {
        double norma;

        int shift = 0;
        for (int i = 0; i < myid; ++i)
            shift += el_num[i] / N;
        shift -= (myid == 0) ? 0 : myid * 2;

        iterations = 0;
        do
        {
            if (send_type == 1)
            {
                //передача вниз
                MPI_Send(y_n.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 1, MPI_COMM_WORLD);
                MPI_Recv(y_n.data(), (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                //передача вверх
                MPI_Send(y_n.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 2, MPI_COMM_WORLD);
                MPI_Recv(y_n.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            if (send_type == 2)
            {
                //передать вниз и принять сверху
                MPI_Sendrecv(y_n.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 3, y_n.data(), (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                //передать вверх и принять снизу
                MPI_Sendrecv(y_n.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 4, y_n.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

            }
            MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
            if (send_type == 3)
            {
                if (myid != np - 1)
                {
                    //передача вниз
                    MPI_Isend(y_n.data() + el_num[myid] - 2 * N, N, MPI_DOUBLE, myid + 1, 5, MPI_COMM_WORLD, &req_send_up);

                    //прием снизу
                    MPI_Irecv(y_n.data() + el_num[myid] - N, N, MPI_DOUBLE, myid + 1, 6, MPI_COMM_WORLD, &req_recv_up);
                }
                if (myid != 0)
                {
                    //прием сверху
                    MPI_Irecv(y_n.data(), N, MPI_DOUBLE, myid - 1, 5, MPI_COMM_WORLD, &req_recv_down);

                    //передача вверх
                    MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, myid - 1, 6, MPI_COMM_WORLD, &req_send_down);
                }
            }

            ++iterations;

            if (send_type == 1 || send_type == 2)
            {
                for (int i = 1; i < el_num[myid] / N - 1; ++i)
                    for (int j = 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;
            }
            if (send_type == 3)
            {
                //все строки, кроме верхней и нижней
                for (int i = 2; i < el_num[myid] / N - 2; ++i)
                    for (int j = 1; j < N - 1; ++j)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;

                if (myid != 0)
                    MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                if (myid != np - 1)
                    MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                //верхняя строка
                int i = 1;
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;

                //нижняя строка
                i = el_num[myid] / N - 2;
                for (int j = 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;
            }

            norma = norm(y, y_n, (myid == 0) ? 0 : N, (myid == np) ? el_num[myid] : el_num[myid] - N);
            MPI_Allreduce(&norma, &norm_f, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            y_n.swap(y);
        } while (norm_f > eps);
    }
    if (myid == 0) {
        if (send_type == 1) {
            std::cout << "Jacobi" << " (MPI_Send + MPI_Recv)\n";
        }
        else if (send_type == 2) {
            std::cout << "Jacobi" << " (MPI_SendRecv)\n";
        }
        else if (send_type == 3) {
            std::cout << "Jacobi" << " (MPI_ISend + MPI_IRecv)\n";
        }
    }
    return norm_f;
}



double Zeidel(std::vector<double>& y, std::vector<double>& y_n, std::vector<int> el_num, int myid, int np, int& iterations, int send_type)
{
    double norm_f;
    //последовательный
    if (np == 1)
    {
        iterations = 0;
        do
        {
            ++iterations;
            for (int i = 1; i < N - 1; ++i)
                for (int j = (i % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f(i * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;

            for (int i = 1; i < N - 1; ++i)
                for (int j = ((i + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f(i * h, j * h) + (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) / coef;

            norm_f = norm(y, y_n, 0, N * N);
            y_n.swap(y);
        } while (norm_f > eps);
    }
    //параллельный
    if (np > 1)
    {
        double norma;

        int shift = 0;
        for (int i = 0; i < myid; ++i)
            shift += el_num[i] / N;
        shift -= (myid == 0) ? 0 : myid * 2;

        iterations = 0;
        do
        {
            if (send_type == 1)
            {
                //передача вниз
                MPI_Send(y_n.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 1, MPI_COMM_WORLD);
                MPI_Recv(y_n.data(), (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                //передача вверх
                MPI_Send(y_n.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 2, MPI_COMM_WORLD);
                MPI_Recv(y_n.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            if (send_type == 2)
            {
                //передать вниз и принять сверху
                MPI_Sendrecv(y_n.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 3, y_n.data(), (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                //передать вверх и принять снизу
                MPI_Sendrecv(y_n.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 4, y_n.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            MPI_Request req_send_up, req_recv_up, req_send_down, req_recv_down;
            if (send_type == 3)
            {

                if (myid != np - 1)
                {
                    //передача вниз
                    MPI_Isend(y_n.data() + el_num[myid] - 2 * N, N, MPI_DOUBLE, myid + 1, 5, MPI_COMM_WORLD, &req_send_up);
                    MPI_Irecv(y_n.data() + el_num[myid] - N, N, MPI_DOUBLE, myid + 1, 6, MPI_COMM_WORLD, &req_recv_up);
                }
                if (myid != 0)
                {
                    //передача вверх
                    MPI_Irecv(y_n.data(), N, MPI_DOUBLE, myid - 1, 5, MPI_COMM_WORLD, &req_recv_down);
                    MPI_Isend(y_n.data() + N, N, MPI_DOUBLE, myid - 1, 6, MPI_COMM_WORLD, &req_send_down);
                }
            }

            ++iterations;

            if (send_type == 1 || send_type == 2)
            {
                for (int i = 1; i < el_num[myid] / N - 1; ++i)
                    for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;
            }
            if (send_type == 3)
            {
                //все строки, кроме верхней и нижней
                for (int i = 2; i < el_num[myid] / N - 2; ++i)
                    for (int j = ((i + shift) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;

                if (myid != 0)
                    MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                if (myid != np - 1)
                    MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                //верхняя строка
                int i = 1;
                for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;

                //нижняя строка
                i = el_num[myid] / N - 2;
                for (int j = ((i + shift) % 2) + 1; j < N - 1; ++j)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y_n[i * N + j - 1] + y_n[i * N + j + 1] + y_n[(i - 1) * N + j] + y_n[(i + 1) * N + j])) / coef;
            }

            if (send_type == 1)
            {
                //передача вниз
                MPI_Send(y.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 1, MPI_COMM_WORLD);
                MPI_Recv(y.data(), (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                //передача вверх
                MPI_Send(y.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 2, MPI_COMM_WORLD);
                MPI_Recv(y.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            if (send_type == 2)
            {
                //передать вниз и принять сверху
                MPI_Sendrecv(y.data() + el_num[myid] - 2 * N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 3, y.data(), (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 3, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

                //передать вверх и принять снизу
                MPI_Sendrecv(y.data() + N, (myid != 0) ? N : 0, MPI_DOUBLE, (myid != 0) ? myid - 1 : np - 1, 4, y.data() + el_num[myid] - N, (myid != np - 1) ? N : 0, MPI_DOUBLE, (myid != np - 1) ? myid + 1 : 0, 4, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            }
            if (send_type == 3)
            {
                if (myid != np - 1)
                {
                    //передача вниз
                    MPI_Isend(y.data() + el_num[myid] - 2 * N, N, MPI_DOUBLE, myid + 1, 5, MPI_COMM_WORLD, &req_send_up);
                    MPI_Irecv(y.data() + el_num[myid] - N, N, MPI_DOUBLE, myid + 1, 6, MPI_COMM_WORLD, &req_recv_up);
                }
                if (myid != 0)
                {
                    //передача вверх
                    MPI_Irecv(y.data(), N, MPI_DOUBLE, myid - 1, 5, MPI_COMM_WORLD, &req_recv_down);
                    MPI_Isend(y.data() + N, N, MPI_DOUBLE, myid - 1, 6, MPI_COMM_WORLD, &req_send_down);
                }
            }

            if (send_type == 1 || send_type == 2)
            {
                for (int i = 1; i < el_num[myid] / N - 1; ++i)
                    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) / coef;
            }
            if (send_type == 3)
            {
                //все строки, кроме верхней и нижней
                for (int i = 2; i < el_num[myid] / N - 2; ++i)
                    for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                        y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) / coef;

                if (myid != 0)
                    MPI_Wait(&req_recv_down, MPI_STATUSES_IGNORE);
                if (myid != np - 1)
                    MPI_Wait(&req_recv_up, MPI_STATUSES_IGNORE);

                //верхняя строка
                int i = 1;
                for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) / coef;

                //нижняя строка
                i = el_num[myid] / N - 2;
                for (int j = (((i + shift) + 1) % 2) + 1; j < N - 1; j += 2)
                    y[i * N + j] = (h * h * f((i + shift) * h, j * h) + (y[i * N + j - 1] + y[i * N + j + 1] + y[(i - 1) * N + j] + y[(i + 1) * N + j])) / coef;

            }

            norma = norm(y, y_n, (myid == 0) ? 0 : N, (myid == np) ? el_num[myid] : el_num[myid] - N);
            MPI_Allreduce(&norma, &norm_f, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            y_n.swap(y);
        } while (norm_f > eps);
    }
    if (myid == 0) {
        if (send_type == 1) {
            std::cout << "Zeidel" << " (MPI_Send + MPI_Recv)\n";
        }
        else if (send_type == 2) {
            std::cout << "Zeidel" << " (MPI_SendRecv)\n";
        }
        else if (send_type == 3) {
            std::cout << "Zeidel" << " (MPI_ISend + MPI_IRecv)\n";
        }
    }
    return norm_f;
}




int main(int argc, char** argv)
{
    int myid, np, iterations;
    double t1, t2, t3, t4, norm_f;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    std::vector<double> y, y_n, y_gen, u;
    std::vector<int> el_num(np), displs(np);


    //распределение размеров блоков
    if (myid == 0)
    {
        if (N % np == 0)
        {
            for (int i = 0; i < np; ++i)
                el_num[i] = (N / np) * N;
        }
        else
        {
            int temp = 0;
            for (int i = 0; i < np - 1; ++i)
            {
                el_num[i] = round(((double)N / (double)np)) * N;
                temp += el_num[i] / N;
            }
            el_num[np - 1] = (N - temp) * N;
        }

        displs[0] = 0;
        for (int i = 1; i < np; ++i)
            displs[i] = displs[i - 1] + el_num[i - 1];

        for (int i = 0; i < np; ++i)
            el_num[i] += 2 * N;
        el_num[0] -= N;
        el_num[np - 1] -= N;
    }
    //рассылка всем процессам
    MPI_Bcast(el_num.data(), np, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), np, MPI_INT, 0, MPI_COMM_WORLD);


    if (myid == 0) {
        std::cout << "np: " << np << std::endl << std::endl;
        y_gen.resize(N * N, 0);
        u.resize(N * N);
        analyt_sol(u);
    }

    if (np == 1) {

        y.resize(el_num[myid], 0);
        zero(y);
        y_n.resize(el_num[myid], 0);
        zero(y_n);

        y.resize(N * N, 0);
        zero(y);
        y_n.resize(N * N, 0);
        zero(y_n);


        t1 = MPI_Wtime();
        norm_f = Jacobi(y, y_n, el_num, myid, np, iterations, 0);
        t2 = MPI_Wtime();
        std::cout << std::endl << "Jacobi seq" << std::endl;
        std::cout << "Time = " << t2 - t1 << std::endl;
        std::cout << "Iterations = " << iterations << std::endl;
        std::cout << "Error = " << norm_f << std::endl;
        std::cout << "|y - u| = " << norm(y, u, 0, N * N) << std::endl << std::endl;

        zero(y);
        zero(y_n);

        t3 = MPI_Wtime();
        norm_f = Zeidel(y, y_n, el_num, myid, np, iterations, 0);
        t4 = MPI_Wtime();
        std::cout << std::endl << "Zeidel seq" << std::endl;
        std::cout << "Time = " << t4 - t3 << std::endl;
        std::cout << "Iterations = " << iterations << std::endl;
        std::cout << "Error = " << norm_f << std::endl;
        std::cout << "|y - u| = " << norm(y, u, 0, N * N) << std::endl << std::endl;
    }
    for (int send_type = 1; send_type <= 3; ++send_type)
    {
        if (np > 1) {
            y.resize(el_num[myid], 0);
            zero(y);
            y_n.resize(el_num[myid], 0);
            zero(y_n);

            t1 = MPI_Wtime();
            norm_f = Jacobi(y, y_n, el_num, myid, np, iterations, send_type);
            t2 = MPI_Wtime();
            if (myid == 0)
            {
                std::cout << "Time = " << t2 - t1 << std::endl;
                std::cout << "Iterations = " << iterations << std::endl;
                std::cout << "Error = " << norm_f << std::endl;
            }
            general_y(el_num, y, y_gen, displs, np, myid);
            if (myid == 0)
                std::cout << "|y - u| = " << norm(y_gen, u, 0, N * N) << std::endl << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);

            y.resize(el_num[myid], 0);
            zero(y);
            y_n.resize(el_num[myid], 0);
            zero(y_n);

            t1 = MPI_Wtime();
            norm_f = Zeidel(y, y_n, el_num, myid, np, iterations, send_type);
            t2 = MPI_Wtime();
            if (myid == 0)
            {
                std::cout << "Time = " << t2 - t1 << std::endl;
                std::cout << "Iterations = " << iterations << std::endl;
                std::cout << "Error = " << norm_f << std::endl;
            }
            general_y(el_num, y, y_gen, displs, np, myid);
            if (myid == 0)
                std::cout << "|y - u| = " << norm(y_gen, u, 0, N * N) << std::endl << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
}