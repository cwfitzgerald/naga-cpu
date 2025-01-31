use std::{
    future::Future,
    mem::MaybeUninit,
    pin::Pin,
    ptr,
    task::{Context, Poll},
};

#[derive(Default)]
struct PendingOnce {
    polled: bool,
}

impl Future for PendingOnce {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.polled {
            Poll::Ready(())
        } else {
            self.polled = true;
            Poll::Pending
        }
    }
}

const WORKGROUP_SIZE: usize = 64;
const SUBGROUP_SIZE: usize = 4;

const SUBGROUPS_PER_WORKGROUP: usize = WORKGROUP_SIZE / SUBGROUP_SIZE;

#[repr(align(32))]
#[derive(Debug, Clone, Copy)]
pub struct Register<T> {
    data: [T; SUBGROUP_SIZE],
}

impl Register<f32> {
    const fn new() -> Self {
        Self {
            data: [0.0_f32; SUBGROUP_SIZE],
        }
    }

    const fn from_array(data: [f32; SUBGROUP_SIZE]) -> Self {
        Self { data }
    }

    fn multiply(self, other: Register<f32>) -> Register<f32> {
        let mut result: Register<f32> = Register::new();
        for i in 0..SUBGROUP_SIZE {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }
}

pub async fn power(
    mut input: Register<f32>,
    multiplier: Register<f32>,
    count: usize,
) -> Register<f32> {
    for _ in 0..count {
        input = input.multiply(multiplier);
        PendingOnce::default().await;
    }
    input
}

#[inline(never)]
pub fn power_for_workgroup(
    inputs: &[f32; WORKGROUP_SIZE],
    multiplier: &[f32; WORKGROUP_SIZE],
    count: usize,
) -> [f32; WORKGROUP_SIZE] {
    let mut result: MaybeUninit<[f32; WORKGROUP_SIZE]> = MaybeUninit::uninit();

    let mut subgroup_futures: [_; SUBGROUPS_PER_WORKGROUP] = std::array::from_fn(|subgroup| {
        power_inner(
            result.as_mut_ptr() as *mut f32,
            inputs,
            multiplier,
            count,
            subgroup,
        )
    });

    let mut finished = false;
    while !finished {
        for subgroup_future in &mut subgroup_futures {
            let pin = unsafe { Pin::new_unchecked(subgroup_future) };

            let result = pin.poll(&mut Context::from_waker(&noop_waker::noop_waker()));

            if let Poll::Ready(()) = result {
                finished = true;
            }
        }
    }

    unsafe { result.assume_init() }
}

async fn power_inner(
    result: *mut f32,
    inputs: &[f32; WORKGROUP_SIZE],
    multiplier: &[f32; WORKGROUP_SIZE],
    count: usize,
    subgroup: usize,
) {
    unsafe {
        let base_invocation = subgroup * SUBGROUP_SIZE;

        let input = Register::from_array(
            inputs
                .get_unchecked(base_invocation..base_invocation + SUBGROUP_SIZE)
                .try_into()
                .unwrap(),
        );

        let multiplier = Register::from_array(
            multiplier
                .get_unchecked(base_invocation..base_invocation + SUBGROUP_SIZE)
                .try_into()
                .unwrap(),
        );

        let output = power(input, multiplier, count).await;

        ptr::copy_nonoverlapping::<f32>(
            output.data.as_ptr(),
            result.add(base_invocation),
            SUBGROUP_SIZE,
        )
    }
}
