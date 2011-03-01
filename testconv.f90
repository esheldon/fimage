! vim: set filetype=fortran et ts=2 sw=2 sts=2 :
function conv_exp_gauss_f4(expIxx,expIxy,expIyy,expXcen,expYcen, &
                           gIxx, gIxy, gIyy,                     &
                           nsub, image, nx, ny) result (flag)

  ! smooth an exponential with a gaussian.  The exp can have
  ! any center.  Place the results in the image.
  !
  ! For a first go, I'm going to simply *add* up at the sub-pixel
  ! level rather than more complex integration
  !
  ! maybe the best way to use this is run it at high resolution
  ! and then downsample

  implicit none

  integer*4 nx,ny
  real*4 image(nx,ny)

  real*4 expIxx,expIxy,expIyy,expXcen,expYcen
  real*4 gIxx, gIxy, gIyy
  real*4 expWx,expWy,expWxy
  real*4 gWx,gWy,gWxy
  real*4 expDet, gDet

  integer*4 ix,iy
  real*4 expExpon, gExpon

  real*4 exp_rng, g_rng, rng
  integer*4 nsub
  integer*4 nstep
  real*4 stepsize
  integer*4 intx, inty
  real*4 x,y,xexp,yexp,xm,ym
  real*4 x2,y2,xm2,ym2
  real*4 csum

  real*4 flag
  flag=0

  expDet = expIxx*expIyy - expIxy**2
  if (expDet .le. 0.) then
    flag=2**0
    return
  endif

  gDet = gIxx*gIyy - gIxy**2
  if (gDet .le. 0.) then
    flag=2**1
    return
  endif

  exp_rng = 7.5*sqrt( max(expIxx,expIyy) )
  g_rng = 4.5*sqrt( max(gIxx,gIyy) )
  rng = max(exp_rng, g_rng)

  expWx=expIxx/expDet
  expWy=expIyy/expDet
  expWxy=expIxy/expDet
  gWx=gIxx/gDet
  gWy=gIyy/gDet
  gWxy=gIxy/gDet

  ! rng is units of pixels, so 
  ! nstep is just twice rng times the number of
  ! subdivisions
  stepsize=1./nsub
  nstep = 2*int(rng)*nsub +1

  print *,"xcen: ",expXcen," ycen: ",expYcen
  print *,"g_rng: ",g_rng," exp_rng: ",exp_rng," rng: ",rng
  print *," nstep: ",nstep," stepsize: ",stepsize


  do ix=1,nx
    do iy=1,ny

      ! integrate over a range -rng,-rng in both
      ! x and y. Use stepsize 1.0/nsub

      ! the gaussian is always centered on this pixel
      ! that is the x,y here starting at -rng

      x = -rng
      x2 = x*x

      ! this is position relative to the exponential
      ! center
      xexp = ix + x
      xm = xexp - expXcen
      xm2 = xm*xm
            
      csum=0
      ! integration steps intx and inty
      do intx=1,nstep
        y = -rng
        y2 = y*y

        yexp = iy+y
        ym = yexp-expYcen
        ym2 = ym*ym

        do inty=1,nstep

          gExpon=0.5*(x2*gWy + y2*gWx - 2.*x*y*gWxy)
          expExpon = xm2*expWy+ym2*expWx-2.*xm*ym*expWxy
          expExpon=sqrt(3.*expExpon)

          csum = csum + exp(-gExpon)*exp(-expExpon)

          y = y+stepsize
          ym = ym+stepsize
          yexp = yexp+stepsize
          y2=y*y
          ym2=ym*ym

        enddo ! integration loop y

        x = x+stepsize
        xm = xm+stepsize
        xexp = xexp+stepsize
        x2=x*x
        xm2=xm*xm

      enddo ! integration loop x
              
      image(ix,iy) = csum

    enddo
  enddo

  return
end function conv_exp_gauss_f4
