use std::{arch::x86_64, ptr};

const WORKGROUP_SIZE: usize = 64;
const SUBGROUP_SIZE: usize = 8;

const SUBGROUPS_PER_WORKGROUP: usize = WORKGROUP_SIZE / SUBGROUP_SIZE;

#[repr(align(32))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Register<T> {
    data: [T; SUBGROUP_SIZE],
}

impl Register<f32> {
    fn new() -> Self {
        Self::default()
    }

    const fn from_array(data: [f32; SUBGROUP_SIZE]) -> Self {
        Self { data }
    }

    fn gather(data: &[f32], offset: Register<u32>) -> Self {
        unsafe {
            let mut result = Register::new();

            for i in 0..(SUBGROUP_SIZE / 8) {
                let offsets =
                    x86_64::_mm256_load_si256(offset.data.as_ptr().add(i * 8) as *const _);
                let length = x86_64::_mm256_set1_epi32(data.len() as i32);

                // 1 if in bounds, 0 if out of bounds
                let cmp = x86_64::_mm256_cmpgt_epi32(length, offsets);
                let mask = x86_64::_mm256_movemask_ps(x86_64::_mm256_castsi256_ps(cmp));

                if mask != 0b11_11_11_11 {
                    panic!("Out of bounds access detected");
                }

                let zeros = x86_64::_mm256_setzero_ps();

                let gathered = x86_64::_mm256_mask_i32gather_ps::<4>(
                    zeros,
                    data.as_ptr(),
                    offsets,
                    x86_64::_mm256_castsi256_ps(cmp),
                );
                x86_64::_mm256_store_ps(result.data.as_mut_ptr().add(i * 8), gathered);
            }

            result
        }
    }

    fn multiply(self, other: Register<f32>) -> Register<f32> {
        let mut result: Register<f32> = Register::new();
        for i in 0..SUBGROUP_SIZE {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }
}

pub fn power(mut input: Register<f32>, multiplier: Register<f32>, count: usize) -> Register<f32> {
    for _ in 0..count {
        input = input.multiply(multiplier);
    }
    input
}

#[inline(never)]
pub fn power_for_workgroup(result: &mut [f32], inputs: &[f32], multiplier: &[f32], count: usize) {
    for subgroup in 0..SUBGROUPS_PER_WORKGROUP {
        power_inner(
            result.as_mut_ptr(),
            TryFrom::try_from(&inputs[0..WORKGROUP_SIZE]).unwrap(),
            TryFrom::try_from(&multiplier[0..WORKGROUP_SIZE]).unwrap(),
            count,
            subgroup,
        );
    }
}

fn power_inner(
    result: *mut f32,
    inputs: &[f32; WORKGROUP_SIZE],
    multiplier: &[f32; WORKGROUP_SIZE],
    count: usize,
    subgroup: usize,
) {
    let base_invocation = subgroup * SUBGROUP_SIZE;

    let input = Register::from_array(
        inputs[base_invocation..base_invocation + SUBGROUP_SIZE]
            .try_into()
            .unwrap(),
    );

    let multiplier = Register::from_array(
        multiplier[base_invocation..base_invocation + SUBGROUP_SIZE]
            .try_into()
            .unwrap(),
    );

    let output = power(input, multiplier, count);

    unsafe {
        ptr::copy_nonoverlapping(
            output.data.as_ptr(),
            result.add(base_invocation),
            SUBGROUP_SIZE,
        )
    };
}

#[inline(never)]
pub fn gather_test(data: &[f32], offset: Register<u32>) -> Register<f32> {
    Register::gather(data, offset)
}
