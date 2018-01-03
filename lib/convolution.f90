! fast convolution routine using fortran 90

module convolution
  
  use omp_lib
  implicit none
  
  real, dimension(2048, 4096) :: iout
  real, dimension(2048, 4096) :: varout
  real, dimension(2048, 4096) :: varout2

contains

  ! set number of threads
  subroutine set_num_threads(n)

    implicit none

    !f2py threadsafe
    !f2py intent(in) n

    INTEGER :: n
    CALL OMP_SET_NUM_THREADS(n)

  end subroutine set_num_threads

  ! convolution
  subroutine conv (nifx, nify, nf, kernel, iin, varin)

    implicit none

    !f2py threadsafe
    !f2py intent(in) nifx, nify, nf
    !f2py intent(in) kernel
    !f2py intent(in) iin, varin

    ! input parameters
    integer, intent(in) :: nifx, nify, nf    ! region to convolve is nifx x nify, kernel is nf x nf
    real*8, dimension(:, :), intent(in) :: kernel
    real, dimension(:, :), intent(in) :: iin, varin  ! input image must be at least (nifx + nf - 1) x (nify + nf - 1)

    ! auxiliar variables
    !integer, target :: k, l
    integer :: k, l

    iout(1: nifx, 1: nify) = 0
    varout(1: nifx, 1: nify) = 0
    varout2(1: nifx, 1: nify) = 0

    ! filter loop
    !$OMP PARALLEL DO private (k, l) shared (nf, nifx, nify, kernel, iin, varin, iout, varout)
    do l = 1, nf
       do k = 1, nf
          iout(1: nifx, 1: nify) &
               = iout(1: nifx, 1: nify) + kernel(k, l) * iin(k: nifx + k, l: nify + l)
          varout(1: nifx, 1: nify) &
               = varout(1: nifx, 1: nify) + kernel(k, l) * kernel(k, l) * varin(k: nifx + k, l: nify + l)
          varout2(1: nifx, 1: nify) &
               = varout2(1: nifx, 1: nify) + kernel(k, l) * varin(k: nifx + k, l: nify + l)
       end do
    end do
    !$OMP END PARALLEL DO

  end subroutine conv

end module convolution
