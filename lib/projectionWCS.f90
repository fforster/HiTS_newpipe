! CC by F.Forster & J.C. Maureira
! fast projection routine using fortran 90
! needs two sets of astrometric solution variables CRPIX, CRVAL, CD, PV and one array with pixel values of the image to be projected (it assumes both images have the same size in pixels)
! the first set of astrometric solution variables are from the reference frame onto which we want to project our new image and the second, from the image to be projected
! TODO: consider variable pixel areas


module projectionWCS
  
  use omp_lib
  implicit none
  
  real*8, parameter :: pi = 3.14159265359
  real*8, parameter :: invpi = 0.31830988618

  ! astrometric solution variables
  integer, dimension(2, 2) :: naxis
  real*8, dimension(2, 2) :: CRPIX
  real*8, dimension(2, 2) :: CRVAL
  real*8, dimension(2, 2, 2) :: CD
  real*8, dimension(2, 2, 2) :: DC
  real*8, dimension(2, 2, 11) :: PV  ! nPV1 = 2, nPV2 = 11
  
  ! radians to degrees and viceversa
  real*8, parameter :: rad2deg = 180. / pi
  real*8, parameter :: deg2rad = pi / 180.

  ! tolerance for Newton method in degrees (1% of one pixel)
  real*8, parameter :: error = 0.0001 * 0.27 / 3600.

  ! output image and variance
  real*8, dimension(2500, 5000) :: imageout, varimageout
  
contains
  
  ! set number of threads
  subroutine set_num_threads(n)
    
    implicit none
    
    !f2py threadsafe
    !f2py intent(in) n
    
    INTEGER :: n
    CALL OMP_SET_NUM_THREADS(n)
    
  end subroutine set_num_threads
  
  ! set header variables
  subroutine setheader(naxisin, CRPIXin, CRVALin, CDin, PVin)
    
    implicit none

    !f2py threadsafe
    !f2py intent(in) naxisin, CRPIXin, CRVALin, CDin, PVin

    integer, intent(in), dimension(2, 2) :: naxisin
    real*8, dimension(2, 2), intent(in) :: CRPIXin
    real*8, dimension(2, 2), intent(in) :: CRVALin
    real*8, dimension(2, 2, 2), intent(in) :: CDin
    real*8, dimension(2, 2, 11), intent(in) :: PVin  ! nPV1 = 2, nPV2 = 11
    real*8 :: invdet
    integer :: k

    naxis = naxisin
    CRPIX = CRPIXin
    CRVAL = CRVALin
    CD = CDin
    PV = PVin
    
    ! compute inverse matrix
    do k = 1, 2
       invdet = 1. / (CD(k, 1, 1) * CD(k, 2, 2) - CD(k, 1, 2) * CD(k, 2, 1))
       DC(k, 1, 1) = CD(k, 2, 2) * invdet
       DC(k, 1, 2) = -CD(k, 1, 2) * invdet
       DC(k, 2, 1) = -CD(k, 2, 1) * invdet
       DC(k, 2, 2) = CD(k, 1, 1) * invdet
    end do

  end subroutine setheader

  ! sinc function (digital processing version)
  real*8 function sinc (x)
    
    implicit none

    real*8 :: x

    if (abs(x) < 1.e-5) then
       sinc = 1.
    else
       sinc = invpi * sin(pi * x) / x 
    endif
    
  end function sinc

  ! convert i, j to x, y
  function ij2xy(i, j, k)

    implicit none

    !f2py threadsafe
    !f2py intent(in) i, j, k
    
    integer, intent(in) :: k
    real*8, intent(in) :: i, j
    real*8, dimension(2) :: ij2xy

    ij2xy(1) = CD(k, 1, 1) * (i - CRPIX(k, 1)) + CD(k, 1, 2) * (j - CRPIX(k, 2))
    ij2xy(2) = CD(k, 2, 1) * (i - CRPIX(k, 1)) + CD(k, 2, 2) * (j - CRPIX(k, 2))

  end function ij2xy

  ! convert xy to xieta
  function xy2xieta(xy, k)

    implicit none
    
    !f2py threadsafe
    !f2py intent(in) xy, k
    
    integer, intent(in) :: k
    real*8, intent(in), dimension(2) :: xy
    real*8 :: x, y, r, x2, x3, y2, y3
    real*8, dimension(2) :: xy2xieta
    
    x = xy(1)
    y = xy(2)
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    r = sqrt(x2 + x2)

    xy2xieta(1) = PV(k, 1, 1) + PV(k, 1, 2) * x + PV(k, 1, 3) * y + PV(k, 1, 4) * r + PV(k, 1, 5) * x2 + PV(k, 1, 6) * x * y + PV(k, 1, 7) * y2 + PV(k, 1, 8) * x3 + PV(k, 1, 9) * x2 * y + PV(k, 1, 10) * x * y2 + PV(k, 1, 11) * y3
    xy2xieta(2) = PV(k, 2, 1) + PV(k, 2, 2) * y + PV(k, 2, 3) * x + PV(k, 2, 4) * r + PV(k, 2, 5) * y2 + PV(k, 2, 6) * y * x + PV(k, 2, 7) * x2 + PV(k, 2, 8) * y3 + PV(k, 2, 9) * y2 * x + PV(k, 2, 10) * y * x2 + PV(k, 2, 11) * x3
    
  end function xy2xieta

  ! convert xieta to RADEC
  function xieta2radec(xieta, k)

    implicit none
    
    !f2py threadsafe
    !f2py intent(in) xieta, k

    integer, intent(in) :: k
    real*8, intent(in), dimension(2) :: xieta
    real*8, dimension(2) :: xieta2radec
    real*8 :: num1, num2, den1, alphap, xi, eta

    xi = xieta(1)
    eta = xieta(2)

    num1 = (xi * deg2rad) / cos(CRVAL(k, 2) * deg2rad) ! rad
    den1 = 1. - (eta * deg2rad) * tan(CRVAL(k, 2) * deg2rad) ! rad
    alphap = atan2(num1, den1) ! rad
    xieta2radec(1)  = CRVAL(k, 1) + alphap * rad2deg ! deg
    num2 = (eta * deg2rad + tan(CRVAL(k, 2) * deg2rad)) * cos(alphap) ! rad
    xieta2radec(2) = atan2(num2, den1) * rad2deg ! deg
    
  end function xieta2radec

  ! convert ij to RADEC
  function radec(i, j, k)

    implicit none
    
    !f2py threadsafe
    !f2py intent(in) i, j, k

    integer, intent(in) :: k
    real*8, intent(in) :: i, j
    real*8, dimension(2) :: xy, xieta, radec

    xy = ij2xy(i, j, k)
    xieta = xy2xieta(xy, k)
    radec = xieta2radec(xieta, k)
    
  end function radec

  ! convert radec to xieta
  function radec2xieta(radec, k)

    implicit none
    
    !f2py threadsafe
    !f2py intent(in) k, radec

    real*8, intent(in), dimension(2) :: radec
    integer, intent(in) :: k
    real*8, dimension(2) :: radec2xieta
    real*8 :: dra, dec, crval2rad
    
    dra = (radec(1) - CRVAL(k, 1)) * deg2rad
    dec = radec(2) * deg2rad
    crval2rad = CRVAL(k, 2) * deg2rad

    radec2xieta(2) = (1. - tan(crval2rad) * cos(dra) / tan(dec)) / &
         (tan(crval2rad) + cos(dra) / tan(dec))

    radec2xieta(1) = tan(dra) * cos(crval2rad) * (1. - radec2xieta(2) * tan(crval2rad))
    
  end function radec2xieta
  
  ! Jacobian [[dxi/dx, dxi/dy], [deta/dx, deta/dy]]
  function xietajac(xy, k)

    implicit none
    
    !f2py threadsafe
    !f2py intent(in) xy, k

    real*8, intent(in), dimension(2) :: xy
    integer, intent(in) :: k
    real*8 :: x, y, r, x2, y2, drdx, drdy
    real*8, dimension(2, 2) :: xietajac
    
    x = xy(1)
    y = xy(2)
    x2 = x * x
    y2 = y * y
    r = sqrt(x2 + x2)

    drdx = x / r
    drdy = y / r

    ! dxi/dx
    xietajac(1, 1) = PV(k, 1, 2) + PV(k, 1, 4) * drdx + PV(k, 1, 5) * 2. * x + PV(k, 1, 6) * y + PV(k, 1, 8) * 3. * x2 + PV(k, 1, 9) * 2. * x * y + PV(k, 1, 10) * y2
    ! dxi/dy
    xietajac(1, 2) = PV(k, 1, 3) + PV(k, 1, 4) * drdy + PV(k, 1, 6) * x + PV(k, 1, 7) * 2. * y + PV(k, 1, 9) * x2 + PV(k, 1, 10) * x * 2. * y + PV(k, 1, 11) * 3. * y2

    ! deta/dx
    xietajac(2, 1) = PV(k, 2, 3) + PV(k, 2, 4) * drdx + PV(k, 2, 6) * y + PV(k, 2, 7) * 2. * x + PV(k, 2, 9) * y2 + PV(k, 2, 10) * y * 2. * x + PV(k, 2, 11) * 3. * x2
    ! deta/dy
    xietajac(2, 2) = PV(k, 2, 2) + PV(k, 2, 4) * drdy + PV(k, 2, 5) * 2. * y + PV(k, 2, 6) * x + PV(k, 2, 8) * 3. * y2 + PV(k, 2, 9) * 2. * y * x + PV(k, 2, 10) * x2
    
  end function xietajac

  ! inverse of Jacobian [[dxi/dx, dxi/dy], [deta/dx, deta/dy]]
  function xietainvjac(xy, k)

    implicit none
    
    !f2py threadsafe
    !f2py intent(in) xy, k

    real*8, intent(in), dimension(2) :: xy
    integer, intent(in) :: k
    real*8 :: x, y, r, x2, y2, drdx, drdy, a, b, c, d, invdet
    real*8, dimension(2, 2) :: xietainvjac
    
    x = xy(1)
    y = xy(2)
    x2 = x * x
    y2 = y * y
    r = sqrt(x2 + x2)

    drdx = x / r
    drdy = y / r

    ! dxi/dx
    a = PV(k, 1, 2) + PV(k, 1, 4) * drdx + PV(k, 1, 5) * 2. * x + PV(k, 1, 6) * y + PV(k, 1, 8) * 3. * x2 + PV(k, 1, 9) * 2. * x * y + PV(k, 1, 10) * y2
    ! dxi/dy
    b = PV(k, 1, 3) + PV(k, 1, 4) * drdy + PV(k, 1, 6) * x + PV(k, 1, 7) * 2. * y + PV(k, 1, 9) * x2 + PV(k, 1, 10) * x * 2. * y + PV(k, 1, 11) * 3. * y2

    ! deta/dx
    c = PV(k, 2, 3) + PV(k, 2, 4) * drdx + PV(k, 2, 6) * y + PV(k, 2, 7) * 2. * x + PV(k, 2, 9) * y2 + PV(k, 2, 10) * y * 2. * x + PV(k, 2, 11) * 3. * x2
    ! deta/dy
    d = PV(k, 2, 2) + PV(k, 2, 4) * drdy + PV(k, 2, 5) * 2. * y + PV(k, 2, 6) * x + PV(k, 2, 8) * 3. * y2 + PV(k, 2, 9) * 2. * y * x + PV(k, 2, 10) * x2

    ! determinant
    invdet = 1. / (a * d - b * c)

    ! inverse matrix (hope that it is invertible)
    xietainvjac(1, 1) = d * invdet
    xietainvjac(1, 2) = -b * invdet
    xietainvjac(2, 1) = -c * invdet
    xietainvjac(2, 2) = a * invdet
    
  end function xietainvjac

  ! invert xieta to xy using Newton method, need inverse jacobian xietainvjac
  function xieta2xy(xieta, k)

    implicit none

    !f2py threadsafe
    !f2py intent(in) xieta, k

    integer, intent(in) :: k
    real*8, intent(in), dimension(2) :: xieta
    real*8, dimension(2) :: xieta2xy
    real*8, dimension(2) :: xietai, dxy
    real*8, dimension(2, 2) :: invjac
    real*8 :: err
    integer :: i
    
    xieta2xy = xieta
    xietai = xy2xieta(xieta2xy, k)
    err = sqrt(dot_product(xieta - xietai, xieta - xietai))
    
    do i = 1, 10 ! maximum of 3 steps
       
       xieta2xy = xieta2xy + matmul(xietainvjac(xieta2xy, k), xieta - xietai)
       xietai = xy2xieta(xieta2xy, k)
       err = sqrt(dot_product(xieta - xietai, xieta - xietai))

       if (err < error) then
          exit
       end if
          
    end do

  end function xieta2xy

  ! compute pixel given xy
  function xy2ij(xy, k)

    implicit none

    !f2py threadsafe
    !f2py intent(in) xy, k

    integer, intent(in) :: k
    real*8, intent(in), dimension(2) :: xy
    real*8, dimension(2) :: xy2ij

    xy2ij = matmul(DC(k, :, :), xy) + CRPIX(k, :)

  end function xy2ij

  ! go from i, j in system k to i, j in system l!
  function ij2ij(i, j, k, l)
    
    implicit none

    !f2py threadsafe
    !f2py intent(in) i, j, k, l

    integer, intent(in) :: k, l
    real*8, intent(in) :: i, j
    real*8, dimension(2) :: ij2ij
    real*8, dimension(2) :: xy, xieta, radec

    xy = ij2xy(i, j, k)
    xieta = xy2xieta(xy, k)
    radec = xieta2radec(xieta, k)
    xieta = radec2xieta(radec, l)
    xy = xieta2xy(xieta * rad2deg, l)
    ij2ij = xy2ij(xy, l)

  end function ij2ij
    
  ! Lanczos interpolation
  subroutine lanczos (alanczos, nx, ny, imagein, varimagein)
    
    implicit none
    
    !f2py threadsafe
    !f2py intent(in) alanczos
    !f2py intent(in) nx, ny, k, l
    !f2py intent(in) imagein

    ! input parameters
    integer, intent(in) :: alanczos
    integer, intent(in) :: nx, ny
    real*8, intent(in), dimension(:, :) :: imagein, varimagein

    ! auxiliar variables
    integer :: i, j, k, l
    integer :: xf, yf
    integer :: xc, yc
    real*8, dimension(2) :: ij
    real*8 :: xi, yi
    real*8 :: xl, Lxl, yl, Lyl, Lxl2Lyl2

    write (*, *) "Lanczos interpolation, npix:", nx, ny

    ! initialize output image
    imageout = 0
    varimageout = 0

    ! Lanczos interpolation
    !$OMP PARALLEL DO private (i, j, k, l, ij, xi, yi, xf, xc, yf, yc, xl, Lxl, yl, Lyl, Lxl2Lyl2) shared(alanczos, nx, ny, imagein, varimagein, imageout, varimageout)

    do j = 1, ny
       do i = 1, nx

          ! transformed pixels
          ij = ij2ij(real(i, 8), real(j, 8), 1, 2)
          xi = ij(1)
          yi = ij(2)
          
          do k = 1 - alanczos, alanczos
             do l = 1 - alanczos, alanczos

                ! we are inside the definition range
                if (xi >= alanczos .and. xi <= nx - alanczos .and. yi >= alanczos .and. yi <= ny - alanczos) then
                   
                   xf = int(floor(xi))
                   xc = int(ceiling(xi))
                   yf = int(floor(yi))
                   yc = int(ceiling(yi))
                   
                   xl = xi - (xf + k)
                   Lxl = sinc(xl) * sinc(xl / alanczos)
                   yl = yi - (yf + l)
                   Lyl = sinc(yl) * sinc(yl / alanczos)
                   Lxl2Lyl2 = Lxl * Lxl * Lyl * Lyl
  
                   imageout(i, j) = imageout(i, j) + imagein(xf + k, yf + l) / 100. * Lxl * Lyl

                   if (varimagein(xf + k, yf + l) > 0) then
                      varimageout(i, j) = varimageout(i, j) + varimagein(xf + k, yf + l) * Lxl2Lyl2
                   end if

                end if
             end do
          end do
       end do
    end do
    !$OMP END PARALLEL DO

    imageout = imageout * 100.

  end subroutine lanczos

end module projectionWCS

