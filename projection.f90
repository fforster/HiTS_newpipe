! fast convolution routine using fortran 90

module projection
  
  use omp_lib
  implicit none

  integer, parameter :: nxmax = 2048
  integer, parameter :: nymax = 4096
  real, dimension(nxmax, nymax)    :: imageout, varimageout, bgout, weights
  integer, dimension(nxmax, nymax) :: dqout
  real, parameter :: pi = 3.14159265359
  real, parameter :: invpi = 0.31830988618
  
contains
  
  ! set number of threads
  subroutine set_num_threads(n)

    implicit none

    !f2py threadsafe
    !f2py intent(in) n

    INTEGER :: n
    CALL OMP_SET_NUM_THREADS(n)
    
  end subroutine set_num_threads

  ! sinc function (digital processing version)
  real function sinc (x)
    
    implicit none

    real :: x

    if (abs(x) < 1.e-6) then
       sinc = 1.
    else
       sinc = invpi * sin(pi * x) / x 
    endif

  end function sinc

  ! Lanczos interpolation
  subroutine lanczos (alanczos, nx, ny, order, astro, imagein, varimagein, dqin, bgin)
    
    implicit none
    
    !f2py threadsafe
    !f2py intent(in) alanczos
    !f2py intent(in) nx, ny
    !f2py intent(in) order
    !f2py intent(in) astro
    !f2py intent(in) imagein, varimagein

    ! input parameters
    integer, intent(in) :: alanczos
    integer, intent(in) :: nx, ny
    integer, intent(in) :: order
    real, intent(in), dimension(:) :: astro
    real, intent(in), dimension(:, :) :: imagein, varimagein, bgin
    integer, intent(in), dimension(:, :) :: dqin

    ! auxiliar variables
    integer :: i, j, k, l, ii, ij, jj, iii, iij, ijj, jjj
    integer :: xf, yf
    integer :: xc, yc
    real :: xi, yi
    real :: xl, Lxl, yl, Lyl, LxlLyl

    write (*, *) nx, ny, astro

    ! initialize output image
    imageout = 0
    varimageout = 0
    weights = 1.

    ! Lanczos interpolation
    !$OMP PARALLEL DO private (k, l, i, j, ii, ij, jj, iii, iij, ijj, jjj, xi, yi, xf, xc, yf, yc, xl, Lxl, yl, Lyl, LxlLyl) shared(order, astro, alanczos, nx, ny, imagein, varimagein, bgin, dqin, imageout, varimageout, bgout, dqout, weights)
    do k = 1 - alanczos, alanczos
       do l = 1 - alanczos, alanczos
          do j = 1, ny
             do i = 1, nx
                
                ! transformed pixels
                xi = astro(1) + astro(3) * i + astro(4) * j
                yi = astro(2) + astro(5) * i + astro(6) * j
                if (order > 1) then
                   ii = i * i
                   ij = i * j
                   jj = j * j
                   xi = xi + astro(7) * ii + astro(8) * ij + astro(9) * jj
                   yi = yi + astro(10) * ii + astro(11) * ij + astro(12) * jj
                   if (order > 2) then
                      iii = ii * i
                      iij = ii * j
                      ijj = i * jj
                      jjj = jj * j
                      xi = xi + astro(13) * iii + astro(14) * iij &
                           + astro(15) * ijj + astro(16) * jjj
                      yi = yi + astro(17) * iii + astro(18) * iij &
                           + astro(19) * ijj + astro(20) * jjj
                   end if
                end if

                ! we are inside the definition range
                if (xi >= alanczos .and. xi <= nx - alanczos .and. yi >= alanczos .and. yi <= ny - alanczos) then
                   
                   xf = int(floor(xi))
                   yf = int(floor(yi))
                   
                   xl = xi - (xf + k)
                   Lxl = sinc(xl) * sinc(xl / alanczos)
                   yl = yi - (yf + l)
                   Lyl = sinc(yl) * sinc(yl / alanczos)
                   LxlLyl = Lxl * Lyl

                   ! image interpolation
                   imageout(i, j) = imageout(i, j) + imagein(xf + k, yf + l) / 100. * LxlLyl

                   ! variance interpolation (note that we don't do proper error propagation, which maked the variance strongly spatially varying
                   varimageout(i, j) = varimageout(i, j) + varimagein(xf + k, yf + l) * LxlLyl !* LxlLyl

                   ! background interpolation
                   bgout(i, j) = bgout(i, j) + bgin(xf + k, yf + l) * LxlLyl
                   
                   ! mask interpolation (we use the largest value)
                   dqout(i, j) = max(dqout(i, j), dqin(xf + k, yf + l))

                   ! for proper normalization (Lanczos doesn't conserve flux)
                   weights(i, j) = weights(i, j) + LxlLyl
                   
                end if
             end do
          end do
       end do
    end do
    !$OMP END PARALLEL DO

    ! normalize Lanczos
    imageout = imageout * 100. / weights
    varimageout = varimageout / weights
    bgout = bgout / weights

  end subroutine lanczos

end module projection
